

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
import torch
import numpy as np

DEADZONE_DEG = 6.5 

# joint order from your earlier print: ['PivotY', 'PivotX']
PIVOT_Y_IDX = 1 #left and right
PIVOT_X_IDX = 0 #up and down

#Helper Functions

ZONE_TO_ACT = {"up": 2, "down": 3, "left": 4, "right": 5, "neutral": 0}
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


 