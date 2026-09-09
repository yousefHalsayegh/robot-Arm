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
    _ensure_progress_tracker
)
from sim.utils.robot_sim import POSITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STICK_DEFAULT_POS = [0.305, -0.058, 0.0]
STICK_DEFAULT_ROT = [0.7071068, 0.0, 0.0, -0.7071068]
def reset_or_restore_on_failure(env, env_ids):
    buf = _ensure_displacement_buffer(env)
    buf[env_ids] = 0.0

    progress_buf = _ensure_progress_tracker(env)
    progress_buf[env_ids] = 0.0

    succeeded = env.termination_manager.get_term("success")[env_ids]
    failed_ids = env_ids[~succeeded]
    # succeeded_ids need no action at all -- leaving the arm/joystick untouched
    # IS "continue from previous position"

    if len(failed_ids) > 0:
        robot = env.scene["robot"]
        obj = env.scene["object"]

        home_target = torch.tensor(
            POSITIONS["home"], device=env.device, dtype=robot.data.joint_pos.dtype
        ).unsqueeze(0).expand(len(failed_ids), -1)
        robot.write_joint_state_to_sim(home_target, torch.zeros_like(home_target), env_ids=failed_ids)

        stick_pos = torch.tensor(STICK_DEFAULT_POS, device=env.device, dtype=obj.data.root_pos_w.dtype)
        stick_rot = torch.tensor(STICK_DEFAULT_ROT, device=env.device, dtype=obj.data.root_quat_w.dtype)
        stick_pose = torch.cat([stick_pos, stick_rot]).unsqueeze(0).expand(len(failed_ids), -1)
        obj.write_root_pose_to_sim(stick_pose, env_ids=failed_ids)

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