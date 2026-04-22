# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#TODO fix up the overal wording to fit the theme of what I am doing more

# def object_is_lifted(
#     env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


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



# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# def object_ee_distance_and_lifted(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Combined reward for reaching the object AND lifting it."""
#     # Get reaching reward
#     reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
#     # Get lifting reward
#     lift_reward = object_is_lifted(env, minimal_height, object_cfg)
#     # Combine rewards multiplicatively
#     return reach_reward * lift_reward
