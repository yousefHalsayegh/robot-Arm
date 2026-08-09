

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
import torch
import numpy as np

DEADZONE_DEG = 6.5 
DISPLACEMENT_THRESHOLD_DEG = 10.0 
# joint order from your earlier print: ['PivotY', 'PivotX']
PIVOT_Y_IDX = 1 #left and right
PIVOT_X_IDX = 0 #up and down

#Helper Functions

ZONE_TO_ACT = {"up": 2, "down": 3, "left": 4, "right": 5, "neutral": 0, "home": 1}
ACT_TO_ZONE = {v: k for k, v in ZONE_TO_ACT.items()}


DEADZONE_DEG = 6.5 

# joint order from your earlier print: ['PivotY', 'PivotX']
PIVOT_Y_IDX = 1
PIVOT_X_IDX = 0

CMD_NEUTRAL = 0
CMD_UP      = 2
CMD_DOWN    = 3
CMD_LEFT    = 4
CMD_RIGHT   = 5
CMD_HOME    = 1

def joystick_zone(object_art, env_index):
    """Classify the joystick's CURRENT physical position into a zone,
    independent of what task it's being driven toward. This is what the
    game should actually see, moment to moment — same as a real controller."""
    tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())  # [PivotY, PivotX]
    y_deg = tilt_deg[PIVOT_Y_IDX]   # left/right
    x_deg = tilt_deg[PIVOT_X_IDX]   # up/down

    if x_deg < -DEADZONE_DEG:
        return "up"
    elif x_deg > DEADZONE_DEG:
        return "down"
    elif y_deg < -DEADZONE_DEG:
        return "left"
    elif y_deg > DEADZONE_DEG:
        return "right"
    return "neutral"

def joystick_registered(object_art, env_index, task):
    tilt = object_art.data.joint_pos[env_index].cpu().numpy()  # [PivotY, PivotX], radians
    tilt_deg = np.rad2deg(tilt)
    x_deg = tilt_deg[PIVOT_X_IDX]   # up/down
    y_deg = tilt_deg[PIVOT_Y_IDX]   # left/right

    if task == "neutral":
        # both axes must be within deadzone
        return abs(x_deg) < DEADZONE_DEG and abs(y_deg) < DEADZONE_DEG
    elif task == "up":
        return x_deg < -DEADZONE_DEG
    elif task == "down":
        return x_deg > DEADZONE_DEG
    elif task == "left":
        return y_deg < -DEADZONE_DEG
    elif task == "right":
        return y_deg > DEADZONE_DEG
    return False

#Reward Functions

def sparse(env, weight=1.0) -> torch.Tensor:
    
    object = env.scene["object"]
    commands = env.command_manager.get_command("joystick_cmd")

    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        task = ACT_TO_ZONE[int(commands[i].item())]
        if joystick_registered(object, i, task):
            reward[i] = weight

    return reward


def step_penalty(env, current_budget=10, weight=1.0) -> torch.Tensor:

    safe_budget = max(current_budget, 1.0)
    penalty     = -(weight / safe_budget)
    # return tensor on correct device
    return torch.full(
        (env.num_envs,), penalty,
        dtype=torch.float32, device=env.device
    )

def axis_bonus(env, weight=1.0) -> torch.Tensor:

    commands = env.command_manager.get_command("joystick_cmd")
    object = env.scene["object"]

    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        cmd  = int(commands[i].item())

        if cmd == CMD_NEUTRAL:
            continue   # no bonus for neutral
 
        tilt_deg = np.rad2deg(
            object.data.joint_pos[i].cpu().numpy()
        )
        x_deg = tilt_deg[PIVOT_X_IDX]
        y_deg = tilt_deg[PIVOT_Y_IDX]
 
        if cmd in (CMD_UP, CMD_DOWN):
            # non-commanded axis is left/right
            if abs(y_deg) < DEADZONE_DEG:
                reward[i] = weight
 
        elif cmd in (CMD_LEFT, CMD_RIGHT):
            # non-commanded axis is up/down
            if abs(x_deg) < DEADZONE_DEG:
                reward[i] = weight
 
    return reward

POSITIONS_HOME = np.array([
    0,      # shoulder_pan
    12,     # shoulder_lift
    53.0,   # elbow_flex
    -63.0,  # wrist_flex
    90.0,   # wrist_roll
    38.717498779296875,  # gripper (open)
], dtype=np.float32)

HOME_TOLERANCE_DEG = 3.0   

def home_reward(env, weight=1.0) -> torch.Tensor:
    """
    Home success: all arm joints (including gripper) within tolerance of the
    home configuration — beside the joystick, gripper open. Unrelated to
    joystick angle entirely.
    """
    commands = env.command_manager.get_command("joystick_cmd")
    robot = env.scene["robot"]
    reward = torch.zeros(env.num_envs, device=env.device)

    home_target_rad = torch.tensor(
        np.deg2rad(POSITIONS_HOME), device=env.device, dtype=torch.float32
    )
    tol_rad = np.deg2rad(HOME_TOLERANCE_DEG)

    for i in range(env.num_envs):
        if int(commands[i].item()) != CMD_HOME:
            continue

        joint_pos = robot.data.joint_pos[i]   # radians, shape [6]
        max_dev = torch.max(torch.abs(joint_pos - home_target_rad))

        if max_dev < tol_rad:
            reward[i] = weight

    return reward

 