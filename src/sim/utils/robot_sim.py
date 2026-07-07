"""
robot_sim.py

Isaac Sim equivalent of robot_noVla.py.
Tracks per-env state (task, action, counters) but does NOT send commands
directly — all articulation calls are batched in game_sim.py.

Joint order matches SO101 USD:
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
"""

import numpy as np
import torch

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# exact values from positions.json converted to radians
POSITIONS = {
    "home": np.array([
        -8.175824175824175,
        16.175824175824175,
        50.46153846153846,
        -75.38461538461539,
        -89.97802197802197,
        60.07731958762886,
    ], dtype=np.float32),

    "neutral": np.array([
        -7.1208791208791204,
        16.615384615384617,
        51.51648351648352,
        -75.38461538461539,
        -89.97802197802197,
        15.811855670103093,
    ], dtype=np.float32),

    "up": np.array([
        -7.1208791208791204,
        20.395604395604394,
        20.92307692307692,
        -75.38461538461539,
        -89.62637362637362,
        15.811855670103093,
    ], dtype=np.float32),

    "down": np.array([
        100.208791208791209,
        100.516483516483516,
        100.03296703296704,
        -75.20879120879121,
        -89.97802197802197,
        15.811855670103093,
    ], dtype=np.float32),
}

for k in POSITIONS:
    POSITIONS[k] = np.deg2rad(POSITIONS[k])

# equivalent to error < 3 degrees on the physical arm
ARRIVAL_THRESHOLD = np.deg2rad(3.0)


class RobotSim:
    """
    Per-env state tracker. Mirrors robot_noVla.py interface.
    Articulation commands are batched externally in game_sim.py.
    """

    def __init__(self, env_index: int):
        self.env_index = env_index
        self.task      = "neutral"
        self.action    = 0
        self.prev      = 0
        self.reseting  = False
        self.actions   = {"all": 0, "neutral": 0, "up": 0, "down": 0}

    def finished(self, articulation) -> bool:
        """
        True when arm has reached current task target.
        Equivalent to error < 3 check in robot_noVla.py.
        """
        current = articulation.data.joint_pos[self.env_index].cpu().numpy()
        target  = POSITIONS[self.task]
        return float(np.abs(current - target).max()) < ARRIVAL_THRESHOLD

    def get_joint_pos(self, articulation) -> np.ndarray:
        """Current joint positions for this env."""
        return articulation.data.joint_pos[self.env_index].cpu().numpy()


def send_targets(articulation, robots, device):
    """
    Batch send joint position targets for all envs at once.
    Call before sim_step() every loop iteration.
    """
    targets = torch.stack([
        torch.tensor(POSITIONS[r.task], dtype=torch.float32)
        for r in robots
    ], dim=0).to(device)

    articulation.set_joint_position_target(targets)
    articulation.write_data_to_sim()


def batch_move_arms(articulation, robots, sim_step_fn,
                    target_name: str, device, n_steps: int = 200):
    """
    Move all arms to target_name simultaneously and wait for arrival.
    Equivalent to robot_noVla.py initial() / reset() but batched.

    Args:
        articulation: Isaac Lab Articulation for the SO101
        robots:       list of RobotSim instances
        sim_step_fn:  callable that advances the sim one step
        target_name:  "home", "neutral", "up", or "down"
        device:       torch device string
        n_steps:      max steps before giving up
    """
    N         = len(robots)
    target_t  = torch.tensor(POSITIONS[target_name], dtype=torch.float32)
    targets   = target_t.unsqueeze(0).repeat(N, 1).to(device)
    confirmed = np.zeros(N, dtype=int)

    for robot in robots:
        robot.reseting = True
        robot.task     = target_name

    for _ in range(n_steps):
        articulation.set_joint_position_target(targets)
        articulation.write_data_to_sim()
        sim_step_fn()

        for i, robot in enumerate(robots):
            current = articulation.data.joint_pos[i].cpu().numpy()
            error   = np.abs(current - POSITIONS[target_name]).max()
            if error < ARRIVAL_THRESHOLD:
                confirmed[i] += 1
            else:
                confirmed[i] = 0

        if (confirmed >= 5).all():
            break

    for robot in robots:
        robot.reseting = False
        robot.action   = 0
        robot.prev     = 0