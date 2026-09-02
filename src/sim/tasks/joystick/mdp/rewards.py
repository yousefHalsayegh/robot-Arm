

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
import torch
import numpy as np
import cv2

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

CMD_TO_TASK  = {
    CMD_NEUTRAL: "neutral", CMD_UP: "up",
    CMD_DOWN: "down", CMD_LEFT: "left", CMD_RIGHT: "right",
    CMD_HOME : "home"
}

BLUE_LOWER  = np.array([100, 80, 50])
BLUE_UPPER  = np.array([130, 255, 255])
WHITE_LOWER = np.array([0, 0, 180])   # confirmed working against the corrected, lit frame
WHITE_UPPER = np.array([180, 60, 255])

MIN_BLOB_AREA = 20

POSITIONS = {
    "home": np.array([
        0,  # shoulder_pan
        12,  # shoulder_lift
        53.0,  # elbow_flex
        -64.0,  # wrist_flex
        90.0,  # wrist_roll
        38.717498779296875,  # gripper
    ], dtype=np.float32),

    "down": np.array([
        -0.4,  # shoulder_pan
        12,  # shoulder_lift
        70.0,  # elbow_flex
        -110.0,  # wrist_flex
        90.0,  # wrist_roll
        6.5,  # gripper
    ], dtype=np.float32),

    "up": np.array([
        -0.4,  # shoulder_pan
        12,  # shoulder_lift
        40.6,  # elbow_flex
        -51.999992,  # wrist_flex
        90.0,  # wrist_roll
        6.5,  # gripper
    ], dtype=np.float32),

    "neutral": np.array([
        -0.4,  # shoulder_pan
        12,  # shoulder_lift
        52.0,  # elbow_flex
        -64.0,  # wrist_flex
        90.0,  # wrist_roll
        6.5,  # gripper
    ], dtype=np.float32),

    "left": np.array([
        5.0,  # shoulder_pan
        12,  # shoulder_lift
        52.0,  # elbow_flex
        -64.0,  # wrist_flex
        75.0,  # wrist_roll
        6.5,  # gripper
    ], dtype=np.float32),

    "right": np.array([
        -5.,  # shoulder_pan
        12,  # shoulder_lift
        52.0,  # elbow_flex
        -64.0,  # wrist_flex
        110.0,  # wrist_roll
        6.5,  # gripper
    ], dtype=np.float32),

    "reset": np.array([
        0,                  # shoulder_pan
        -97.40282517223994,   # shoulder_lift
        91.67324722093173,    # elbow_flex
        -85.94366926962348,   # wrist_flex
        90.0,    # wrist_roll
        0.0,                  # gripper
    ], dtype=np.float32),
}

HOME_TOLERANCE_DEG = 3.0   
GRIPPER_WRIST_DEG = 5.0 
OVERSHOOT_MARGIN_DEG = 3.0  
OVERSHOOT_DECAY_DEG  = 10.0 
def _largest_contour_centroid(mask, min_area=MIN_BLOB_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _closest_point_to_target(mask, target_xy, min_area=MIN_BLOB_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    pts = largest.reshape(-1, 2).astype(np.float32)
    dists = np.linalg.norm(pts - np.array(target_xy, dtype=np.float32), axis=1)
    idx = np.argmin(dists)
    return tuple(pts[idx])


def _detect_joystick_and_arm(rgb_uint8: np.ndarray):
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    blue_mask  = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    joystick_xy = _largest_contour_centroid(blue_mask)
    arm_xy = None
    if joystick_xy is not None:
        arm_xy = _closest_point_to_target(white_mask, joystick_xy)

    return joystick_xy, arm_xy


def _ensure_progress_tracker(env):
    if not hasattr(env, "_joystick_best_progress"):
        env._joystick_best_progress = torch.zeros(env.num_envs, device=env.device)
    return env._joystick_best_progress


def reset_progress_tracker(env, env_ids):
    """EventCfg reset term — zero the tracker for envs being reset."""
    buf = _ensure_progress_tracker(env)
    buf[env_ids] = 0.0
    

def joystick_zone(object_art, env_index):

    tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())  # [PivotY, PivotX]
    y_deg = tilt_deg[PIVOT_Y_IDX]   # left/right
    x_deg = tilt_deg[PIVOT_X_IDX]   # up/down

    if x_deg < -DEADZONE_DEG:
        return "up"
    elif x_deg > DEADZONE_DEG:
        return "down"
    elif y_deg > DEADZONE_DEG:
        return "left"
    elif y_deg < -DEADZONE_DEG:
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
        return y_deg > DEADZONE_DEG
    elif task == "right":
        return y_deg < -DEADZONE_DEG
    return False

#Reward Functions

def sparse(env, weight=1.0) -> torch.Tensor:
    
    object = env.scene["object"]
    commands = env.command_manager.get_command("joystick_cmd")

    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        task = ACT_TO_ZONE[int(commands[i].item())]
        tilt_deg = np.rad2deg(object.data.joint_pos[i].cpu().numpy())
        x_deg, y_deg = tilt_deg[PIVOT_X_IDX], tilt_deg[PIVOT_Y_IDX]
        print(f"[sparse] env={i} task={task} x_deg={x_deg:.2f} y_deg={y_deg:.2f}")
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
    object_art = env.scene["object"]
    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        cmd = int(commands[i].item())
        task = ACT_TO_ZONE.get(cmd, "neutral")
        if task not in ("up", "down", "left", "right"):
            continue

        if not joystick_registered(object_art, i, task):
            continue  # only evaluate cleanliness AT the moment of real success

        tilt_deg = np.rad2deg(object_art.data.joint_pos[i].cpu().numpy())
        x_deg, y_deg = tilt_deg[PIVOT_X_IDX], tilt_deg[PIVOT_Y_IDX]
        print(f"[axis_bonus] env={i} task={task} x_deg={x_deg:.2f} y_deg={y_deg:.2f}")
        if task in ("up", "down"):
            if abs(y_deg) < DEADZONE_DEG:
                reward[i] = weight
        else:  # left, right
            if abs(x_deg) < DEADZONE_DEG:
                reward[i] = weight

    return reward

def vision_shaping_reward(
    env,
    weight_center:   float = 0.3,
    weight_approach: float = 1.0,
) -> torch.Tensor:

    camera = env.scene["side"]
    rgb_batch = camera.data.output["rgb"].cpu().numpy() 

    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        frame = rgb_batch[i]
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx**2 + cy**2)

        joystick_xy, arm_xy = _detect_joystick_and_arm(frame)

        if arm_xy is not None:
            ax, ay = arm_xy
            center_dist = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
            reward[i] += weight_center * (1.0 - min(center_dist / max_dist, 1.0))

            if joystick_xy is not None:
                jx, jy = joystick_xy
                approach_dist = np.sqrt((ax - jx) ** 2 + (ay - jy) ** 2)
                reward[i] += weight_approach * (1.0 - min(approach_dist / max_dist, 1.0))
       

    return reward

def joystick_progress_reward(env, weight: float = 0.5) -> torch.Tensor:
    commands = env.command_manager.get_command("joystick_cmd")
    object_art = env.scene["object"]
    best_progress = _ensure_progress_tracker(env)
    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        cmd = int(commands[i].item())
        task = ACT_TO_ZONE.get(cmd, "neutral")
        if task not in ("up", "down", "left", "right"):
            continue

        tilt_deg = torch.rad2deg(object_art.data.joint_pos[i])
        x_deg, y_deg = tilt_deg[PIVOT_X_IDX], tilt_deg[PIVOT_Y_IDX]

        if task == "up":
            progress = -x_deg
        elif task == "down":
            progress = x_deg
        elif task == "left":
            progress = y_deg
        else:
            progress = -y_deg

        # map raw progress -> the same ramp/plateau/decay shape as before,
        # but only pay for IMPROVEMENT over the episode's best mark so far
        print(f"[progress] env={i} task={task} x_deg={x_deg:.2f} y_deg={y_deg:.2f}")
        if progress <= 0:
            shaped = 0.0
        elif progress < DEADZONE_DEG:
            shaped = (progress / DEADZONE_DEG).item()
        elif progress <= DEADZONE_DEG + OVERSHOOT_MARGIN_DEG:
            shaped = 1.0
        else:
            overshoot = progress - (DEADZONE_DEG + OVERSHOOT_MARGIN_DEG)
            shaped = max(0.0, 1.0 - (overshoot / OVERSHOOT_DECAY_DEG).item())

        delta = max(0.0, shaped - best_progress[i].item())
        reward[i] = weight * delta
        best_progress[i] = max(best_progress[i].item(), shaped)

    return reward