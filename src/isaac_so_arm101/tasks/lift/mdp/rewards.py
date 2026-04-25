

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, Camera 
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#camera help
def _get_images(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    
    camera: Camera = env.scene[camera_cfg.name]
    rgb   = camera.data.output.get("rgb",   None)
    depth = camera.data.output.get("distance_to_image_plane", None)
    if rgb is None or depth is None:
        return None, None
    depth_f = depth[..., 0].clone()
    depth_f[~torch.isfinite(depth_f)] = 999.0
    return rgb[..., :3], depth_f

def _find_joystick_camera_space(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    num_envs, H, W, _ = rgb.shape
    device = rgb.device
 
    r = rgb[..., 0].float()
    g = rgb[..., 1].float()
    b = rgb[..., 2].float()

    blue_mask = (b > r + 20) & (b > g + 20) & (b > 100)
    valid_depth = (depth > 0.05) & (depth < 10.0)
    blue_mask   = blue_mask & valid_depth
 
    found = blue_mask.any(dim=-1).any(dim=-1)
    mask_f     = blue_mask.float()
    depth_safe = depth.clamp(min=0.01)
    weights    = mask_f / depth_safe            # (N, H, W)
    total      = weights.sum(dim=(-2,-1)).clamp(min=1e-6)  # (N,)
 
    u_grid = torch.arange(W, device=device).float().view(1, 1, W)
    v_grid = torch.arange(H, device=device).float().view(1, H, 1)
 
    # Weighted centroid pixel
    u_joy = (weights * u_grid).sum(dim=(-2,-1)) / total   # (N,)
    v_joy = (weights * v_grid).sum(dim=(-2,-1)) / total   # (N,)
    

    u_idx = u_joy.long().clamp(0, W-1)
    v_idx = v_joy.long().clamp(0, H-1)
    env_idx = torch.arange(rgb.shape[0], device=device)
    # Weighted mean depth — biased toward shallowest blue pixels
    d_at_centroid = depth[env_idx, v_idx, u_idx]
    depth_masked = torch.where(blue_mask, depth, torch.full_like(depth, 999.0))
    d_min        = depth_masked.reshape(num_envs, -1).min(dim=-1).values
    centroid_valid = (d_at_centroid > 0.05) & (d_at_centroid < 10.0)
    d_joy = torch.where(centroid_valid, d_at_centroid, d_min)

    print(f"u_joy={u_joy[0]:.1f} v_joy={v_joy[0]:.1f} d_joy={d_joy[0]:.3f}m")
    print(f"blue pixels found: {blue_mask[0].sum().item()}")
 
    # Unproject to camera-space 3D
    fx = K[:, 0, 0];  fy = K[:, 1, 1]
    cx = K[:, 0, 2];  cy = K[:, 1, 2]

    X = (u_joy - cx) / fx * d_joy
    Y = (v_joy - cy) / fy * d_joy
    Z = d_joy
 
    point_3d = torch.stack([X, Y, Z], dim=-1)   # (N, 3)
    # Zero out not-found envs
    point_3d = point_3d * found.float().unsqueeze(-1)
    return point_3d, found

def _camera_to_world_space(
    points_cam: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:

    #this seem to assume that it knows the camera position in relation to the world, which I might need ot chnage or set to make it easier
    cam_pos  = camera.data.pos_w        # (N, 3)
    cam_quat = camera.data.quat_w_ros   # (N, 4) w,x,y,z
 
    # Build R^T (world → camera rotation)
    w = cam_quat[:, 0:1]; x = cam_quat[:, 1:2]
    y = cam_quat[:, 2:3]; z = cam_quat[:, 3:4]
 
    # Row vectors of rotation matrix (world → camera = R^T)
    # R maps camera→world, so R^T maps world→camera
    Rt0 = torch.cat([1-2*(y*y+z*z),  2*(x*y+w*z),   2*(x*z-w*y)], dim=-1)  # (N,3)
    Rt1 = torch.cat([2*(x*y-w*z),    1-2*(x*x+z*z), 2*(y*z+w*x)], dim=-1)
    Rt2 = torch.cat([2*(x*z+w*y),    2*(y*z-w*x),   1-2*(x*x+y*y)], dim=-1)
 
    # Translate then rotate        # (N, 3)
    px = (points_cam * Rt0).sum(dim=-1)
    py = (points_cam * Rt1).sum(dim=-1)
    pz = (points_cam * Rt2).sum(dim=-1)
    
    return torch.stack([px, py, pz], dim=-1) + cam_pos # (N, 3)


#sensor help 
def _get_gripper_position_from_joints(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    ee_frame : FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    return ee_w


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
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    
    camera : Camera = env.scene[camera_cfg.name]
    rgb, depth = _get_images(env, camera_cfg)

    K = camera.data.intrinsic_matrices


    joy_cam, joy_found = _find_joystick_camera_space(rgb, depth, K)
    gripper_w   = _get_gripper_position_from_joints(env, ee_frame_cfg)
    joy_w = _camera_to_world_space(joy_cam, camera)

    print(f"Detected joystick in camera space: {joy_cam[0]}")
    print(f"Expected:                          (0.070, 0.020, 0.923)")
    print(f"Camera pos: {camera.data.pos_w[0]}")

    # After computing joy_w:
    print(f"Recovered joystick world pos: {joy_w[0]}")
    print(f"Expected:                     (0.377, 0.07, 0.08)")
    distance = (joy_w - gripper_w).norm(dim=-1)   # (N,)
    reward   = 1.0 - torch.tanh(distance / std)

    return torch.where(joy_found, reward, torch.zeros_like(reward))


def touch_joystick(
    env: ManagerBasedRLEnv,
    touch: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    
    camera: Camera = env.scene[camera_cfg.name]
    rgb, depth = _get_images(env, camera_cfg)

 
    K = camera.data.intrinsic_matrices
 
    joy_cam,    joy_found  = _find_joystick_camera_space(rgb, depth, K)
    gripper_w              = _get_gripper_position_from_joints(env, ee_frame_cfg)
    joy_w            = _camera_to_world_space(joy_cam, camera)
 
    distance = (joy_w - gripper_w).norm(dim=-1)
 
    return (joy_found & (distance < touch)).float()