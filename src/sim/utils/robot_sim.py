
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
        2.5,  # shoulder_pan
        12,  # shoulder_lift
        53.0,  # elbow_flex
        -63.0,  # wrist_flex
        91.0,  # wrist_roll
        38.717498779296875,  # gripper
    ], dtype=np.float32),

    "down": np.array([
        2.5,  # shoulder_pan
        10,  # shoulder_lift
        62.0,  # elbow_flex
        -72.0,  # wrist_flex
        91.0,  # wrist_roll
        9.05,  # gripper
    ], dtype=np.float32),

    "up": np.array([
        2.5,  # shoulder_pan
        13,  # shoulder_lift
        46.0,  # elbow_flex
        -56.0,  # wrist_flex
        91.0,  # wrist_roll
        9.05,  # gripper
    ], dtype=np.float32),

    "neutral": np.array([
        2.5,  # shoulder_pan
        12,  # shoulder_lift
        52.0,  # elbow_flex
        -62.0,  # wrist_flex
        91.0,  # wrist_roll
        9.05,  # gripper
    ], dtype=np.float32),

    "reset": np.array([
        1.0,                  # shoulder_pan
        -97.40282517223994,   # shoulder_lift
        91.67324722093173,    # elbow_flex
        -85.94366926962348,   # wrist_flex
        91.67324722093173,    # wrist_roll
        0.0,                  # gripper
    ], dtype=np.float32),

}

for k in POSITIONS:
    POSITIONS[k] = np.deg2rad(POSITIONS[k])

# equivalent to error < 3 degrees on the physical arm
ARRIVAL_THRESHOLD = np.deg2rad(1.5)


class RobotSim:
    """
    Per-env state tracker. Mirrors robot_noVla.py interface.
    Articulation commands are batched externally in game_sim.py.
    """

    def __init__(self, env_index: int):
        self.env_index = env_index
        self.task      = "neutral"
        self.reseting  = False
        self.actions   = {"all": 0, "neutral": 0, "up": 0, "down": 0}
        self.moving      = False   # True while transitioning to a newly-decided action
        self.pending_act = 0 

    def finished(self, articulation, max_steps=100) -> bool:
        """
        True when arm has reached current task target.
        Equivalent to error < 3 check in robot_noVla.py.
        """
        current = articulation.data.joint_pos[self.env_index].cpu().numpy()
        target  = POSITIONS[self.task]
        per_joint_error = np.abs(current - target)
        error = float(per_joint_error.max())

        if error < ARRIVAL_THRESHOLD:
            self._step_count = 0  # reset for next task
            return True

        if max_steps is not None:
            self._step_count = getattr(self, "_step_count", 0) + 1
            if self._step_count > max_steps:
                joint_names = articulation.joint_names
                still_over = [
                    (joint_names[j], float(per_joint_error[j]))
                    for j in range(len(joint_names))
                    if per_joint_error[j] >= ARRIVAL_THRESHOLD
                ]
                print(f"[env {self.env_index}] task '{self.task}' not reached after "
                      f"{max_steps} steps — joints still over threshold: {still_over}")

        return False

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
    joint_names = articulation.joint_names

    for robot in robots:
        robot.reseting = True
        robot.task     = target_name
    print("moving ", target_name)
    for _ in range(n_steps):
        articulation.set_joint_position_target(targets)
        articulation.write_data_to_sim()
        sim_step_fn()

        for i, robot in enumerate(robots):
            current = articulation.data.joint_pos[i].cpu().numpy()
            per_joint_error   = np.abs(current - POSITIONS[target_name]).max()
            error      = per_joint_error.max()

            if error < ARRIVAL_THRESHOLD:
                
                confirmed[i] += 1
            else:
                confirmed[i] = 0
        
        if (confirmed >= 5).all():
            break
    for i, robot in enumerate(robots):
        current = articulation.data.joint_pos[i].cpu().numpy()
        per_joint_error = np.abs(current - POSITIONS[target_name])
        still_over = [
            (joint_names[j], float(per_joint_error[j]))
            for j in range(len(joint_names))
            if per_joint_error[j] >= ARRIVAL_THRESHOLD
        ]
        if still_over:
            print(f"robot {i} — joints still not under threshold: {still_over}")

    for robot in robots:
        robot.reseting = False
        robot.action   = 0
        robot.prev     = 0

def batch_move_arm(articulation, robot, sim_step_fn,
                    target_name: str, device, n_steps: int = 200):
    
    target_t  = torch.tensor(POSITIONS[target_name], dtype=torch.float32)
    targets   = target_t.unsqueeze(0).to(device)
    confirmed = 0
    joint_names = articulation.joint_names

    robot.reseting = True
    robot.task     = target_name

    for _ in range(n_steps):
        articulation.set_joint_position_target(targets)
        articulation.write_data_to_sim()
        sim_step_fn()

        current = articulation.data.joint_pos.cpu().numpy()
        per_joint_error   = np.abs(current - POSITIONS[target_name]).max()
        error      = per_joint_error.max()

        if error < ARRIVAL_THRESHOLD:
            
            confirmed += 1
        else:
            confirmed = 0
        
        if (confirmed >= 5):
            break


    robot.reseting = False
    robot.action   = 0
    robot.prev     = 0