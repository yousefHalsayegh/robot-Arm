

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

def joystick_zone(object_art, env_index):

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

def home_reward(env, weight=1.0) -> torch.Tensor:

    commands = env.command_manager.get_command("joystick_cmd")
    robot = env.scene["robot"]
    reward = torch.zeros(env.num_envs, device=env.device)

    home_target_rad = torch.tensor(
        np.deg2rad(POSITIONS["home"]), device=env.device, dtype=torch.float32
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


def gripper_wrist_shaping_reward(env, weight: float = 1.0) -> torch.Tensor:

    commands = env.command_manager.get_command("joystick_cmd")
    robot = env.scene["robot"]
    reward = torch.zeros(env.num_envs, device=env.device)
    scale_rad = np.deg2rad(GRIPPER_WRIST_DEG)

    for i in range(env.num_envs):
        cmd = int(commands[i].item())
        task = CMD_TO_TASK[cmd]
        if task not in POSITIONS:
            continue

        target = torch.tensor(POSITIONS[task], device=env.device, dtype=robot.data.joint_pos.dtype)
        joint_pos = robot.data.joint_pos[i]

        gripper_error = torch.abs(joint_pos[5] - target[5])
        wrist_error = torch.abs(joint_pos[4] - target[4])

        gripper_reward = torch.exp(-gripper_error / scale_rad)   # bounded (0,1], 1 at perfect match
        wrist_reward = torch.exp(-wrist_error / scale_rad)

        reward[i] = weight * (gripper_reward + wrist_reward) / 2.0

    return reward