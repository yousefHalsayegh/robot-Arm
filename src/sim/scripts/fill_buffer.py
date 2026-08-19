import argparse
from isaaclab.app import AppLauncher

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Buffer pre-fill for low-level joystick SAC")
    parser.add_argument("--task",              type=str, default=None)
    parser.add_argument("--num_envs",          type=int, default=1)
    parser.add_argument("--disable_fabric",    action="store_true", default=False)
    parser.add_argument("--lerobot_repo_id",   type=str, default=None,
                        help="HuggingFace repo id of existing LeRobot dataset")
    parser.add_argument("--output_path",       type=str, default="buffer_prefill.pkl")
    parser.add_argument("--synthetic_per_cmd", type=int, default=500,
                        help="synthetic trajectories per command")
    parser.add_argument("--interp_steps",      type=int, default=50,
                        help="interpolation steps for synthetic trajectories")
    parser.add_argument("--decision_steps", type=int, default=100,
                         help="physics steps per frozen-action window — must match live DECISION_STEPS")
    parser.add_argument("--action_scale_deg", type=float, default=5.0,
                         help="max per-step joint delta in degrees — must match live ACTION_SCALE")

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher   = AppLauncher(args_cli)
    simulation_app = app_launcher.app


import pickle
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import glob
import os
import gymnasium as gym
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import wandb
from sim.utils.robo_brain import ReplayBuffer
from sim.tasks.joystick.mdp.observations import update_frame_stack, Frames
from sim.tasks.joystick.mdp.rewards import (
    joystick_registered,
    CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT, CMD_HOME
)
from sim.utils.robot_sim import POSITIONS

import sim.tasks.joystick.play_env_cfg as _cfg_module  # noqa: F401 — force module load

import ale.config as config
 
ALL_COMMANDS = [CMD_LEFT, CMD_HOME, CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_RIGHT]
CMD_TO_TASK  = {
    CMD_NEUTRAL: "neutral", CMD_UP: "up",
    CMD_DOWN: "down", CMD_LEFT: "left", CMD_RIGHT: "right",
    CMD_HOME : "home"
}
TASK_CHAIN = {
    "home":    ["reset", "home"],
    "neutral": ["reset", "home", "neutral"],
    "up":      ["neutral", "up"],
    "down":    ["neutral", "down"],
    "left":    ["neutral", "left"],   
    "right":   ["neutral", "right"],  
}
TASK_INDEX_TO_CMD = {
    0: CMD_UP,    
    1: CMD_NEUTRAL,
    2: CMD_NEUTRAL,  
    3: CMD_DOWN,     

}

HOP_TASK_NAME = {
    ("reset", "home"): "go to home",
    ("home", "neutral"): "start",
    ("neutral", "up"): "go to up",
    ("neutral", "down"): "go to down",
    ("neutral", "left"): "go to left",
    ("neutral", "right"): "go to right",
}

HOME_TOLERANCE_DEG = 0.3
HOP_ARRIVAL_TOLERANCE_DEG = 1.5
MAX_WINDOWS_PER_HOP = 30

def get_newest_rgb_frame(cam_state_uint8: np.ndarray) -> np.ndarray:

    return np.stack([cam_state_uint8[12], cam_state_uint8[13], cam_state_uint8[14]], axis=-1)  # [224,224,3]

def is_warmup_hop(hop_idx: int, chain_len: int) -> bool:

    return hop_idx != (chain_len - 2)

def _home_registered(robot, env_index: int) -> bool:
    """Mirrors home_reward's arrival check: all arm joints (incl. gripper)
    within tolerance of POSITIONS['home']."""
    joint_pos = robot.data.joint_pos[env_index].cpu().numpy()
    home_target_rad = POSITIONS["home"]
    tol_rad = np.deg2rad(HOME_TOLERANCE_DEG)
    return bool(np.max(np.abs(joint_pos - home_target_rad)) < tol_rad)

def _thin_waypoints(frames: list[np.ndarray], min_deg: float = 1.0) -> list[np.ndarray]:
    """Keep only frames that differ from the last kept frame by at least
    min_deg on some joint — collapses near-duplicate consecutive recorded
    frames into meaningful waypoints. Always keeps the final frame."""
    if not frames:
        return frames
    min_rad = np.deg2rad(min_deg)
    kept = [frames[0]]
    for f in frames[1:-1]:
        if np.max(np.abs(f - kept[-1])) >= min_rad:
            kept.append(f)
    kept.append(frames[-1])
    return kept

def sim_step(base_env, simulation_app):
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    simulation_app.update()
 
 
def set_arm_joints(base_env, joint_positions: np.ndarray, device: str):
    """Teleport arm to exact joint positions instantly."""
    robot = base_env.scene["robot"]
    target = torch.tensor(
        joint_positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    zeros = torch.zeros_like(target)
    env_ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(target, zeros, env_ids=env_ids)
 
 
def capture_transition(
    base_env, frame_stack, joint_pos, action_delta, reward,
    next_joint, done, n_steps, simulation_app, device,
    cam_state=None,  # NEW: pass in cached render if available
):
    if cam_state is None:
        set_arm_joints(base_env, joint_pos, device)
        sim_step(base_env, simulation_app)
        update_frame_stack(base_env, [frame_stack])
        cam_state = (frame_stack._get_state() * 255).round().astype(np.uint8)

    set_arm_joints(base_env, next_joint, device)
    sim_step(base_env, simulation_app)
    update_frame_stack(base_env, [frame_stack])
    cam_next = (frame_stack._get_state() * 255).round().astype(np.uint8)

    return (
        cam_state, joint_pos.copy(), action_delta.copy(), reward,
        cam_next, next_joint.copy(), float(done), n_steps,
    ), cam_next  # return cam_next separately so caller can cache it
 
# ── Source 1 — LeRobot physical demonstrations ────────────────────────────────
def load_lerobot_demos(repo_id: str, local_dir: str = None) -> list[dict]:
    try:


        parquet_pattern = os.path.join(
            local_dir or f"/home/yousef/.cache/huggingface/lerobot/{repo_id}/",
            "data/chunk-000", "*.parquet"
        )
        files = sorted(glob.glob(parquet_pattern))
        if not files:
            raise FileNotFoundError(f"no parquet files found at {parquet_pattern}")

        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"loaded {len(df)} frames from parquet")
        print(f"columns: {list(df.columns)}")
        

        # group by episode
        episodes = {}
        for _, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx not in episodes:
                episodes[ep_idx] = []
            episodes[ep_idx].append(row.to_dict())

        result = [episodes[k] for k in sorted(episodes.keys())]
        print(f"loaded {len(result)} episodes")
        return result

    except Exception as e:
        print(f"could not load LeRobot dataset: {e}")
        return []
 
def infer_command_from_episode(frames: list[dict]) -> int | None:
    """
    Look up the command directly from task_index, using the real mapping
    from this dataset's tasks.parquet. Returns None if the episode's
    task_index has no known mapping (caller should skip it).
    """
    if not frames:
        return None
    task_idx = int(frames[0]["task_index"])

    return TASK_INDEX_TO_CMD.get(task_idx, None)
 
 
def convert_lerobot_to_buffer(
    base_env, frame_stacks, buffer, repo_id, device, simulation_app,
    decision_steps, action_scale, gamma,
):
    episodes = load_lerobot_demos(repo_id)
    if not episodes:
        print("no LeRobot episodes found — skipping")
        return

    fs    = frame_stacks[0]
    robot = base_env.scene["robot"]
    joystick_term = base_env.command_manager.get_term("joystick_cmd")
    total_pushed = 0

    with tqdm(total=len(episodes), desc="LeRobot demos") as pbar:
        for ep_idx, episode in enumerate(episodes):
            if len(episode) < 2:
                pbar.update(1)
                continue

            command = infer_command_from_episode(episode)
            if command is None:
                print(f"episode {ep_idx}: unmapped task_index — skipping")
                pbar.update(1)
                continue
            task = CMD_TO_TASK[command]


            if task not in POSITIONS:
                print(f"episode {ep_idx}: task '{task}' not in POSITIONS — skipping (can't align)")
                pbar.update(1)
                continue

            # per-episode offset: align this episode's final recorded frame
            # to POSITIONS[task], apply the same additive offset throughout
            last_frame_state = np.deg2rad(np.array(episode[-1]["observation.state"], dtype=np.float32))

            
            offset = POSITIONS[task] - last_frame_state


            aligned_frames = [
                np.deg2rad(np.array(f["observation.state"], dtype=np.float32)) + offset
                for f in episode
            ]

            joystick_term._command[0] = command


            # teleport directly to the episode's own (aligned) first frame —
            # no scripted home hop in front of it
            set_arm_joints(base_env, aligned_frames[0], device)
            sim_step(base_env, simulation_app)
            update_frame_stack(base_env, [fs], reset_ids=[0])

            joint_pos_current = aligned_frames[0].copy()
            aligned_frames = _thin_waypoints(aligned_frames)
            n_hops = len(aligned_frames) - 1
            for hop_idx in range(n_hops):
                target = aligned_frames[hop_idx + 1]

                cam_state = (fs._get_state() * 255).round().astype(np.uint8)

                raw_delta = (target - joint_pos_current) / decision_steps
                action = np.clip(raw_delta, -action_scale, action_scale)

                target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)

                discounted_reward = 0.0
                env_ids = torch.tensor([0], device=device)
                for pd_step in range(decision_steps):
                    robot.set_joint_position_target(target_t,  env_ids=env_ids)
                    robot.write_data_to_sim()
                    sim_step(base_env, simulation_app)
                    step_reward_tensor = base_env.reward_manager.compute(dt=base_env.step_dt)
                    discounted_reward += (gamma ** pd_step) * float(step_reward_tensor[0].item())

                joint_pos_next = robot.data.joint_pos[0].cpu().numpy()

                update_frame_stack(base_env, [fs])
                cam_next = (fs._get_state() * 255).round().astype(np.uint8)

                is_last_hop = (hop_idx == n_hops - 1)
                done = joystick_registered(base_env.scene["object"], 0, task) if is_last_hop else False

                buffer.push(
                    cam_state, joint_pos_current, action.copy(), discounted_reward,
                    cam_next, joint_pos_next.copy(), float(done), decision_steps,
                )
                total_pushed += 1
                joint_pos_current = joint_pos_next

            pbar.set_postfix({
                "ep": ep_idx, "cmd": task, "hops": n_hops,
                "buffer": len(buffer), "pushed": total_pushed,
            }, refresh=True)
            pbar.update(1)

    print(f"LeRobot conversion done — buffer size: {len(buffer)}")
 
 
# ── Source 2 — Synthetic POSITIONS transitions ────────────────────────────────
 
def generate_synthetic_transitions(
    base_env, frame_stacks, buffer, brain, steps_counter,
    n_per_command, decision_steps, action_scale, gamma, device, simulation_app,
    args_wandb=False, train_every_n_pushes=10,
    export_lerobot=False   
):
    fs = frame_stacks[0]
    robot = base_env.scene["robot"]
    joystick_term = base_env.command_manager.get_term("joystick_cmd")
    total_pushed = 0
    successful_hops = 0   

    lerobot_dataset = None
    if export_lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id="local/joystick_synthetic",
            fps=int(1.0 / (decision_steps * base_env.sim.get_physics_dt())),
            root="/home/yousef/.cache/huggingface/lerobot/",
            features={
                "observation.state": {"dtype": "float32", "shape": (6,),
                    "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]},
                "action": {"dtype": "float32", "shape": (6,),
                    "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]},
                "observation.images.side": {"dtype": "video", "shape": (224, 224, 3),
                    "names": ["height", "width", "channels"]},
            },
        )

    valid_tasks = [(cmd, task, chain) for cmd in ALL_COMMANDS
                   for task in [CMD_TO_TASK[cmd]] if task in POSITIONS
                   for chain in [TASK_CHAIN.get(task)]
                   if chain and all(k in POSITIONS for k in chain)]

    schedule = []
    for cmd, task, chain in valid_tasks:
        schedule.extend([(cmd, task, chain)] * n_per_command)
    np.random.shuffle(schedule)

    per_task_traj_counter = {task: 0 for _, task, _ in valid_tasks}

    with tqdm(total=len(schedule), desc="Synthetic transitions", dynamic_ncols=True) as pbar:
        for cmd, task, chain in schedule:
            per_task_traj_counter[task] += 1
            traj_idx = per_task_traj_counter[task]

            joystick_term._command[0] = cmd
            start = POSITIONS[chain[0]] + np.deg2rad(np.random.uniform(-1.0, 1.0, size=len(POSITIONS[chain[0]])))
            set_arm_joints(base_env, start, device)
            sim_step(base_env, simulation_app)
            update_frame_stack(base_env, [fs], reset_ids=[0])
            joint_pos_current = start.copy()

            for hop_idx in range(len(chain) - 1):
                target_key = chain[hop_idx + 1]
                target = POSITIONS[target_key]
                warmup = is_warmup_hop(hop_idx, len(chain))

                cam_state = (fs._get_state() * 255).round().astype(np.uint8)
                raw_delta = (target - joint_pos_current) / decision_steps
                action = np.clip(raw_delta, -action_scale, action_scale)
                target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)

                discounted_reward = 0.0
                for step_idx in range(decision_steps):
                    robot.set_joint_position_target(target_t, env_ids=torch.tensor([0], device=device))
                    robot.write_data_to_sim()
                    sim_step(base_env, simulation_app)
                    step_reward_tensor = base_env.reward_manager.compute(dt=base_env.step_dt)
                    discounted_reward += (gamma ** step_idx) * float(step_reward_tensor[0].item())

                joint_pos_next = robot.data.joint_pos[0].cpu().numpy()
                update_frame_stack(base_env, [fs])
                cam_next = (fs._get_state() * 255).round().astype(np.uint8)

                is_final_hop = (hop_idx == len(chain) - 2)
                done = (_home_registered(robot, 0) if task == "home"
                        else joystick_registered(base_env.scene["object"], 0, task)) if is_final_hop else False

                if not warmup:
                    buffer.push(cam_state, joint_pos_current, action.copy(), discounted_reward,
                                cam_next, joint_pos_next.copy(), float(done), decision_steps)
                    total_pushed += 1
                    if done:
                        successful_hops += 1   # NEW

                    if total_pushed % train_every_n_pushes == 0:
                        critic_loss, actor_loss, train_diagnostics = brain.train()
                        steps_counter[0] += 1
                        if args_wandb:
                            wandb.log({
                                "pretrain/critic_loss": critic_loss,
                                "pretrain/actor_loss": actor_loss,
                                "pretrain/alpha": brain.alpha.item(),
                                "pretrain/successful_hops": successful_hops,   
                                "pretrain/success_rate": successful_hops / max(total_pushed, 1),  
                                **{f"pretrain/{k.split('/')[-1]}": v for k, v in train_diagnostics.items()},
                            }, step=steps_counter[0])

                    if lerobot_dataset is not None:
                        task_str = HOP_TASK_NAME.get((chain[hop_idx], target_key), f"go to {target_key}")
                        lerobot_dataset.add_frame({
                            "observation.state": np.rad2deg(joint_pos_current).astype(np.float32),
                            "observation.images.side": get_newest_rgb_frame(cam_next),
                            "action": np.rad2deg(target).astype(np.float32),
                            "task": task_str,
                        })

                joint_pos_current = joint_pos_next

            if lerobot_dataset is not None:
                lerobot_dataset.save_episode()

            pbar.set_postfix({"cmd": task, "traj": f"{traj_idx}/{n_per_command}",
                              "pushed": total_pushed, "successes": successful_hops}, refresh=True)
            pbar.update(1)

    if lerobot_dataset is not None:
        lerobot_dataset.finalize()

    print(f"synthetic generation done — buffer size: {len(buffer)}, "
          f"successful hops: {successful_hops}/{total_pushed}")
 

 
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,   # single env for pre-filling
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
 
    base_env = env.unwrapped
    device   = str(base_env.device)
 
    # warm up sim
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    simulation_app.update()
 
    env.reset()
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
 
    # frame stacks
    frame_stacks = [Frames(n=4)]
    update_frame_stack(base_env, frame_stacks, reset_ids=[0])
 
    # buffer
    buffer = ReplayBuffer(capacity=int(config.CAPACITY))
 
    # ── Source 1 — LeRobot physical demos ─────────────────────────────────────
    if args_cli.lerobot_repo_id:
        convert_lerobot_to_buffer(
            base_env, frame_stacks, buffer,
            repo_id=args_cli.lerobot_repo_id,
            device=device,
            simulation_app=simulation_app,
            decision_steps=args_cli.decision_steps,
            action_scale=np.deg2rad(args_cli.action_scale_deg),
            gamma=config.GAMMA,
        )
    else:
        print("no lerobot_repo_id provided — skipping LeRobot conversion")
 
    # ── Source 2 — Synthetic POSITIONS ────────────────────────────────────────
    generate_synthetic_transitions(
            base_env, frame_stacks, buffer,
            n_per_command=args_cli.synthetic_per_cmd,
            decision_steps=args_cli.decision_steps,
            action_scale=np.deg2rad(args_cli.action_scale_deg),
            gamma=config.GAMMA,
            device=device,
            simulation_app=simulation_app,
        )
 
    # ── Save buffer ───────────────────────────────────────────────────────────
    print(f"saving buffer ({len(buffer)} transitions) to {args_cli.output_path}")
    with open(args_cli.output_path, "wb") as f:
        pickle.dump(buffer, f)
    print("done")
 
    env.close()
 
 
if __name__ == "__main__":
    main()
    simulation_app.close()
