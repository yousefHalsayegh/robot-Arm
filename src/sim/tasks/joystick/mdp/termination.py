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

    result = torch.zeros(env.num_envs, device=env.device)

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
            result[i] = registered and bool(max_disp[i] > DISPLACEMENT_THRESHOLD_DEG)
        else:
            result[i] = joystick_registered(object_art, i, task)

    return result