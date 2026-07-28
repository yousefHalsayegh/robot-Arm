
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
        0,  # shoulder_pan
        12,  # shoulder_lift
        53.0,  # elbow_flex
        -63.0,  # wrist_flex
        90.0,  # wrist_roll
        38.717498779296875,  # gripper
    ], dtype=np.float32),

    "down": np.array([
        -0.4,  # shoulder_pan
        10,  # shoulder_lift
        57.0,  # elbow_flex
        -69.0,  # wrist_flex
        90.0,  # wrist_roll
        8.5,  # gripper
    ], dtype=np.float32),

    "up": np.array([
        -0.4,  # shoulder_pan
        12,  # shoulder_lift
        45.6,  # elbow_flex
        -60.999992,  # wrist_flex
        90.0,  # wrist_roll
        8.5,  # gripper
    ], dtype=np.float32),

    "neutral": np.array([
        -0.4,  # shoulder_pan
        12,  # shoulder_lift
        52.0,  # elbow_flex
        -63.0,  # wrist_flex
        90.0,  # wrist_roll
        8.5,  # gripper
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


POSITIONS_T = {k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in POSITIONS.items()}

def send_targets(articulation, robots):
    targets = torch.stack([POSITIONS_T[r.task] for r in robots], dim=0)
    articulation.set_joint_position_target(targets)
    articulation.write_data_to_sim()



def batch_move_arms(articulation, robots, sim_step_fn,
                    target_name: str, device,  n_steps: int = 200):
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

    for _ in range(80):
        sim_step_fn()

    for robot in robots:
        robot.reseting = False
        robot.action   = 0
        robot.prev     = 0

def batch_move_arm(articulation, robot, sim_step_fn,
                    target_name: str, device, n_steps: int = 200):
    
    target_t  = torch.tensor(POSITIONS[target_name], dtype=torch.float32)
    targets   = target_t.unsqueeze(0).to(device)
    confirmed = 0

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

def calibrate_zone_position(articulation, object_art, env_index,
                            direction, sim_step_fn,
                            pivot_x_idx, deadzone_deg,
                            POSITIONS,
                            margin_deg=0.2,
                            elbow_step_frac=0.1,
                            wrist_step_frac=0.05,
                            settle_steps=10,
                            convergence_max_steps=150,
                            neutral_settle_steps=30):
    neutral = POSITIONS["neutral"]
    full_target = POSITIONS[direction]
 
    elbow_delta = full_target[2] - neutral[2]
    wrist_delta = full_target[3] - neutral[3]
    wrist_sign = -1.0 if direction == "up" else 1.0
 
    elbow_fracs = np.arange(elbow_step_frac, 1.0 + 1e-9, elbow_step_frac)
    wrist_fracs = np.arange(wrist_step_frac, 1.0 + 1e-9, wrist_step_frac)
 
    def build_candidate(f_elbow, f_wrist):
        candidate = full_target.copy()
        candidate[2] = neutral[2] + f_elbow * elbow_delta
        candidate[3] = neutral[3] + wrist_sign * f_wrist * wrist_delta
        return candidate
 
    def crossed_with_margin(axis_deg):
        return (axis_deg < -(deadzone_deg + margin_deg)) if direction == "up" \
            else (axis_deg > (deadzone_deg + margin_deg))
 
    def crossed_plain(axis_deg):
        return (axis_deg < -deadzone_deg) if direction == "up" \
            else (axis_deg > deadzone_deg)
 
    # --- Phase 1: coarse sweep, one representative candidate per elbow row ---
    representative_candidates = []
    for f_elbow in elbow_fracs:
        row_found = None
        for f_wrist in wrist_fracs:
            candidate = build_candidate(f_elbow, f_wrist)
            target_t = torch.tensor(candidate, dtype=torch.float32, device="cuda").unsqueeze(0)
            articulation.set_joint_position_target(target_t)
            articulation.write_data_to_sim()
 
            for _ in range(settle_steps):
                sim_step_fn()
 
            tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())
            axis_deg = tilt_deg[pivot_x_idx]
            crossed = crossed_with_margin(axis_deg)
 
            print(f"[{direction}] f_elbow={f_elbow:.2f} f_wrist={f_wrist:.2f} "
                  f"tilt={axis_deg:.2f} crossed={crossed}")
 
            if crossed and row_found is None:
                row_found = candidate.copy()
                print(f"[{direction}] row f_elbow={f_elbow:.2f}: first crossing at "
                      f"f_wrist={f_wrist:.2f} -- will speed-test this one")
 
        if row_found is not None:
            representative_candidates.append((f_elbow, row_found))
 
    if not representative_candidates:
        print(f"[{direction}] no elbow row ever crossed deadzone+margin -- "
              f"check deadzone_deg/margin_deg, or whether the full target "
              f"is even large enough to reach it.")
        return full_target
 
    # --- Phase 2: speed-test each representative candidate from a fresh neutral start ---
    neutral_t = torch.tensor(neutral, dtype=torch.float32, device="cuda").unsqueeze(0)
 
    def reset_to_neutral():
        for _ in range(neutral_settle_steps):
            articulation.set_joint_position_target(neutral_t)
            articulation.write_data_to_sim()
            sim_step_fn()
 
    best_candidate = None
    best_steps = None
 
    for f_elbow, candidate in representative_candidates:
        reset_to_neutral()
 
        target_t = torch.tensor(candidate, dtype=torch.float32, device="cuda").unsqueeze(0)
        steps_taken = None
        for step in range(1, convergence_max_steps + 1):
            articulation.set_joint_position_target(target_t)
            articulation.write_data_to_sim()
            sim_step_fn()
 
            tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())
            axis_deg = tilt_deg[pivot_x_idx]
 
            if crossed_plain(axis_deg):
                steps_taken = step
                break
 
        status = f"crossed in {steps_taken} steps" if steps_taken is not None \
            else f"never crossed within {convergence_max_steps} steps"
        print(f"[{direction}] speed test f_elbow={f_elbow:.2f}: {status}")
 
        if steps_taken is not None and (best_steps is None or steps_taken < best_steps):
            best_steps = steps_taken
            best_candidate = candidate
 
    reset_to_neutral()  # leave the arm in a clean state when done
 
    if best_candidate is None:
        print(f"[{direction}] none of the representative candidates converged "
              f"within {convergence_max_steps} steps when commanded directly -- "
              f"falling back to the full target.")
        return full_target
 
    print(f"[{direction}] fastest-converging candidate: {best_steps} steps")
    return best_candidate
 




def measure_joystick_response(articulation, object_art, robot, sim_step_fn,
                               target_name, pivot_x_idx, deadzone_deg, physics_dt,
                               max_steps=300):
    target_t = torch.tensor(POSITIONS[target_name], dtype=torch.float32, device="cuda").unsqueeze(0)
    for step in range(max_steps):
        articulation.set_joint_position_target(target_t)
        articulation.write_data_to_sim()
        sim_step_fn()

        tilt_deg = np.rad2deg(object_art.data.joint_pos[robot.env_index].cpu().numpy())
        axis_deg = tilt_deg[pivot_x_idx]
        print(tilt_deg)
        crossed = (axis_deg < -deadzone_deg) if target_name == "up" else (axis_deg > deadzone_deg)

        if crossed:
            print(f"{target_name}: joystick crossed deadzone in {step+1} steps = {(step+1)*physics_dt:.3f}s")

            return step + 1

    print(f"{target_name}: never crossed deadzone within {max_steps} steps")
    return None

def calibrate_shoulder_pan_centering(articulation, object_art, env_index, sim_step_fn,
                                      POSITIONS,
                                      lateral_axis_idx=0,
                                      step_deg=1.0,
                                      settle_steps=30,
                                      home_settle_steps=30,
                                      tolerance=0.003,
                                      max_iters=25, 
                                      grip_settle_steps = 80):

    base_shoulder_pan = POSITIONS["neutral"][0]
    home_t = torch.tensor(POSITIONS["home"], dtype=torch.float32, device="cuda").unsqueeze(0)
 
    def measure_lateral(offset):
        # reset to a clean, gripper-open state before EVERY trial, so
        # each offset is evaluated independently rather than inheriting
        # slip/contact history from whichever offset was tested before it
        for _ in range(home_settle_steps):
            articulation.set_joint_position_target(home_t)
            articulation.write_data_to_sim()
            sim_step_fn()
 
        candidate = POSITIONS["neutral"].copy()
        candidate[0] = base_shoulder_pan + offset
        target_t = torch.tensor(candidate, dtype=torch.float32, device="cuda").unsqueeze(0)
 
        # phase 1: let the arm's joints reach the commanded target
        for _ in range(settle_steps):
            articulation.set_joint_position_target(target_t)
            articulation.write_data_to_sim()
            sim_step_fn()
 
        # phase 2: keep holding the same target and let the JOYSTICK's own
        # contact-induced motion settle -- the arm reaching its target
        # doesn't mean the gripped object has stopped moving yet
        for _ in range(grip_settle_steps):
            articulation.set_joint_position_target(target_t)
            articulation.write_data_to_sim()
            sim_step_fn()
 
        loc = object_art.data.joint_pos[env_index].cpu().numpy()
        return loc[lateral_axis_idx]
 
    step = np.deg2rad(step_deg)
    best_offset = 0.0
    best_abs = abs(measure_lateral(0.0))
    print(f"[centering] start: offset=0.00deg lateral={best_abs:.5f}")
 
    iters = 0
    while step > np.deg2rad(0.05) and iters < max_iters and best_abs >= tolerance:
        improved = False
        for direction in (1.0, -1.0):
            candidate_offset = best_offset + direction * step
            lateral = measure_lateral(candidate_offset)
            print(f"[centering] try offset={np.rad2deg(candidate_offset):.2f}deg "
                  f"lateral={lateral:.5f}")
            if abs(lateral) < best_abs:
                best_abs = abs(lateral)
                best_offset = candidate_offset
                improved = True
                break  # take the improving direction, continue from there
 
        iters += 1
        if not improved:
            step /= 2.0
 
    # validate the chosen offset with one more fresh transition -- confirms
    # it actually reproduces what the search measured, rather than trusting
    # a value that may itself have been the last in a contaminated chain
    confirmed_lateral = measure_lateral(best_offset)
    print(f"[centering] final offset={np.rad2deg(best_offset):.2f}deg, "
          f"search-measured lateral={best_abs:.5f}, "
          f"confirmed (fresh) lateral={confirmed_lateral:.5f} "
          f"(after {iters} iterations)")
 
    if abs(confirmed_lateral) > tolerance * 3:
        print(f"[centering] WARNING: confirmed lateral ({confirmed_lateral:.5f}) is "
              f"notably worse than the search's own estimate -- the search may "
              f"still be unreliable (e.g. contact instability at this pose). "
              f"Consider lowering step_deg or investigating before trusting this offset.")
 
    for key in ("neutral", "up", "down"):
        POSITIONS[key][0] += best_offset
 
    return best_offset
