from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from sim.tasks.joystick.mdp.rewards import (
    joystick_registered,
    CMD_TO_TASK,
    PIVOT_X_IDX,
    PIVOT_Y_IDX,
    DISPLACEMENT_THRESHOLD_DEG,
    HOME_TOLERANCE_DEG,
)
from sim.utils.robot_sim import POSITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_or_restore_on_failure(env, env_ids):
    buf = _ensure_displacement_buffer(env)
    buf[env_ids] = 0.0

    succeeded = env.termination_manager.get_term("success")[env_ids]
    failed_ids = env_ids[~succeeded]
    succeeded_ids = env_ids[succeeded]

    if len(failed_ids) > 0:
        robot = env.scene["robot"]
        obj = env.scene["object"]
        target_arm = env._safe_arm_joint_pos[failed_ids]
        robot.write_joint_state_to_sim(target_arm, torch.zeros_like(target_arm), env_ids=failed_ids)
        obj.write_root_pose_to_sim(env._safe_joystick_pose[failed_ids], env_ids=failed_ids)

    if len(succeeded_ids) > 0:
        env._safe_arm_joint_pos[succeeded_ids] = env.scene["robot"].data.joint_pos[succeeded_ids].clone()
        obj = env.scene["object"]
        env._safe_joystick_pose[succeeded_ids] = torch.cat(
            [obj.data.root_pos_w[succeeded_ids], obj.data.root_quat_w[succeeded_ids]], dim=-1
        )

def _ensure_displacement_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Lazily create the per-env joystick-displacement tracker."""
    if not hasattr(env, "_joystick_max_displacement"):
        env._joystick_max_displacement = torch.zeros(env.num_envs, device=env.device)
    return env._joystick_max_displacement


def reset_displacement_tracker(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """EventCfg reset term — zero the tracker for the envs being reset.
    Registered with mode='reset', same lifecycle stage as randomise_controller_pose."""
    buf = _ensure_displacement_buffer(env)
    buf[env_ids] = 0.0


def success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:

    commands = env.command_manager.get_command("joystick_cmd")
    object_art = env.scene["object"]
    robot = env.scene["robot"]
    max_disp = _ensure_displacement_buffer(env)

    result = torch.zeros(env.num_envs, dtype=torch.bool,  device=env.device)

    for i in range(env.num_envs):
        cmd = int(commands[i].item())
        task = CMD_TO_TASK.get(cmd, "neutral")

        tilt_deg = torch.rad2deg(object_art.data.joint_pos[i, [PIVOT_X_IDX, PIVOT_Y_IDX]])
        max_disp[i] = torch.maximum(max_disp[i], tilt_deg.abs().max())

        if task == "home":
            joint_pos = robot.data.joint_pos[i]
            home_target = torch.tensor(POSITIONS["home"], device=env.device, dtype=joint_pos.dtype)
            result[i] = torch.max(torch.abs(joint_pos - home_target)) < np.deg2rad(HOME_TOLERANCE_DEG)
        elif task == "neutral":
            registered = joystick_registered(object_art, i, task)
            result[i] = bool(registered) and bool(max_disp[i] > DISPLACEMENT_THRESHOLD_DEG)
        else:
            result[i] = bool(joystick_registered(object_art, i, task))

    return result