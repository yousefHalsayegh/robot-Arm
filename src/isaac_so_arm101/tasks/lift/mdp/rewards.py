

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, Camera 
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#helper
def _centroid_of_color(
    rgb: torch.Tensor,
    target_rgb: tuple[float, float, float],
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the pixel centroid of a colour in the image.
 
    Args:
        rgb:        (num_envs, H, W, 3) float32 0-1
        target_rgb: Target colour (r, g, b) in 0-1
        threshold:  L1 colour distance tolerance
 
    Returns:
        centroid_u: (num_envs,) horizontal pixel coordinate (or -1 if not found)
        centroid_v: (num_envs,) vertical   pixel coordinate (or -1 if not found)
    """
    num_envs, H, W, _ = rgb.shape
    device = rgb.device
 
    target = torch.tensor(target_rgb, device=device, dtype=torch.float32)
    dist   = (rgb - target).abs().sum(dim=-1)   # (num_envs, H, W)
    mask   = (dist < threshold).float()          # (num_envs, H, W)
 
    # Pixel coordinate grids
    u_grid = torch.arange(W, device=device).float().view(1, 1, W).expand(num_envs, H, W)
    v_grid = torch.arange(H, device=device).float().view(1, H, 1).expand(num_envs, H, W)
 
    total = mask.sum(dim=(-1, -2)).clamp(min=1e-6)  # (num_envs,)
 
    centroid_u = (mask * u_grid).sum(dim=(-1, -2)) / total  # (num_envs,)
    centroid_v = (mask * v_grid).sum(dim=(-1, -2)) / total
 
    # Mark envs where colour was not found at all
    found = mask.sum(dim=(-1, -2)) > 0   # (num_envs,)
    centroid_u = torch.where(found, centroid_u, torch.full_like(centroid_u, -1.0))
    centroid_v = torch.where(found, centroid_v, torch.full_like(centroid_v, -1.0))
 
    return centroid_u, centroid_v


# The below is the direct way
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object(target) using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3). the joystick
    target_pos_w = ee_frame.data.target_pos_w[..., 1, :]
    # End-effector position: (num_envs, 3). The robot
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(target_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(distance / std)

def graping_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
     
    """Reward the agent for getting the accuator close to the target with a specfic threshold"""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = ee_frame.data.target_pos_w[..., 1, :]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos_w - ee_w, dim=1)

    return (distance < threshold).float()

def grap_and_hold_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    """Reward the agent for getting the accuator close to the target with a specfic threshold and closing the gripper.
    so this take into consideration the gripper position"""
    
    # simialr to before with the grasping bit
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = ee_frame.data.target_pos_w[..., 1, :]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos_w - ee_w, dim=1)
    compare = distance < threshold

    #the closing check
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.find_joints("gripper")[0]
    pos = robot.data.joint_pos[:,idx].squeeze(-1)
    closed = pos < 0.1

    return (compare & closed).float()

# The camera way 


def joystick_reach_reward(
    env: ManagerBasedRLEnv,
    std: float,
    rob_rgb : set,
    joy_rgb : set,
    joy_threshold: float, 
    rob_threshold: float,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    
    camera: Camera               = env.scene[camera_cfg.name]
 

    #data from the camera 
    rgb = camera.data.output.get("rgb", None)

    #joystick centroid 
    joy_u, joy_v = _centroid_of_color(
        rgb, joy_rgb, joy_threshold
    )

    #robot/gripper centroid 
    rob_u, rob_v = _centroid_of_color(
        rgb, rob_rgb, rob_threshold
    )

    du = joy_u - rob_u
    dv = joy_v - rob_v
    pixel_dist = torch.sqrt(du * du + dv * dv)

    visible = (joy_u >= 0) & (rob_u >= 0)
    reward = 1.0 - torch.tanh(pixel_dist / std)

    return torch.where(visible, reward, torch.zeros_like(reward))


def touch_joystick(
    env: ManagerBasedRLEnv,
    touch: float,
    rob_rgb : set,
    joy_rgb : set,
    joy_threshold: float, 
    rob_threshold: float,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        
)-> torch.Tensor:
    camera: Camera               = env.scene[camera_cfg.name]

    #data from the camera 
    rgb = camera.data.output.get("rgb", None)
    

    #joystick centroid 
    joy_u, joy_v = _centroid_of_color(
        rgb, joy_rgb, joy_threshold
    )

    #robot/gripper centroid 
    rob_u, rob_v = _centroid_of_color(
        rgb, rob_rgb, rob_threshold
    )

    visible = (joy_u >= 0) & (rob_u >= 0)
    
    du = joy_u - rob_u
    dv = joy_v - rob_v
    pixel_dist = torch.sqrt(du * du + dv * dv)
    
    return (visible & (pixel_dist < touch)).float()