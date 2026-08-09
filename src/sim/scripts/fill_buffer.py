import argparse
from isaaclab.app import AppLauncher
 
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
 
from sim.utils.robo_brain import ReplayBuffer
from sim.tasks.joystick.mdp.observations import update_frame_stack, Frames
from sim.tasks.joystick.mdp.rewards import (
    joystick_registered,
    CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT,CMD_HOME
)
from sim.utils.robot_sim import POSITIONS

import sim.tasks.joystick.play_env_cfg as _cfg_module  # noqa: F401 — force module load
 
import ale.config as config
 
ALL_COMMANDS = [CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT, CMD_HOME]
CMD_TO_TASK  = {
    CMD_NEUTRAL: "neutral", CMD_UP: "up",
    CMD_DOWN: "down", CMD_LEFT: "left", CMD_RIGHT: "right",
    CMD_HOME : "home"
}


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

    robot.write_joint_state_to_sim(target, zeros)
 
 
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
 
def infer_command_from_episode(frames: list[dict]) -> int:
    """
    Infer the joystick command from the episode task label or
    joint trajectory direction. Returns CMD_* integer.
 
    LeRobot frames contain 'task' or 'task_index' fields.
    """
    if not frames:
        return CMD_HOME
 
    # try task label first
    first = frames[0]
    if "task" in first:
        task_str = str(first["task"]).lower()
        if "up" in task_str:
            return CMD_UP
        elif "down" in task_str:
            return CMD_DOWN
        elif "left" in task_str:
            return CMD_LEFT
        elif "right" in task_str:
            return CMD_RIGHT
        elif "neutral" in task_str:
                return CMD_NEUTRAL
        else:
            return CMD_HOME
 
    # fallback — infer from joint trajectory direction
    # compare first and last joint positions
    if "observation.state" in first:
        start = np.array(first["observation.state"])
        end   = np.array(frames[-1]["observation.state"])
        delta = end - start
        # largest delta joint determines direction
        # this is a heuristic — update based on your SO101 joint mapping
        if abs(delta[1]) > 0.1:   # shoulder_lift
            return CMD_UP if delta[1] > 0 else CMD_DOWN
 
    return CMD_NEUTRAL
 
 
def convert_lerobot_to_buffer(
    base_env,
    frame_stacks: list,
    buffer:       ReplayBuffer,
    repo_id:      str,
    device:       str,
    simulation_app,
):
    episodes = load_lerobot_demos(repo_id)
    if not episodes:
        print("no LeRobot episodes found — skipping")
        return

    fs           = frame_stacks[0]
    total_pushed = 0

    with tqdm(total=len(episodes), desc="LeRobot demos") as pbar:

        for ep_idx, episode in enumerate(episodes):
            if len(episode) < 2:
                pbar.update(1)
                continue

            command = infer_command_from_episode(episode)
            task    = CMD_TO_TASK.get(command, "neutral")

            joint_0 = np.array(episode[0]["observation.state"])
            set_arm_joints(base_env, joint_0, device)
            sim_step(base_env, simulation_app)
            update_frame_stack(base_env, [fs], reset_ids=[0])

            n_steps = len(episode) - 1

            for step_idx in range(len(episode) - 1):
                frame      = episode[step_idx]
                frame_next = episode[step_idx + 1]

                joint_pos    = np.array(frame["observation.state"])
                joint_next   = np.array(frame_next["observation.state"])
                action_delta = joint_next - joint_pos

                done   = joystick_registered(
                    base_env.scene["object"], 0, task
                ) if step_idx == len(episode) - 2 else False
                reward = 1.0 if done else 0.0

                transition = capture_transition(
                    base_env, fs,
                    joint_pos, action_delta, reward,
                    joint_next, done,
                    n_steps - step_idx,
                    simulation_app, device,
                )
                buffer.push(*transition)
                total_pushed += 1

            pbar.set_postfix({
                "ep":     ep_idx,
                "cmd":    task,
                "frames": n_steps,
                "done":   done,
                "buffer": len(buffer),
                "pushed": total_pushed,
            }, refresh=True)
            pbar.update(1)

    print(f"LeRobot conversion done — buffer size: {len(buffer)}")
 
 
# ── Source 2 — Synthetic POSITIONS transitions ────────────────────────────────
 
def generate_synthetic_transitions(
    base_env,
    frame_stacks:  list[Frames],
    buffer:        ReplayBuffer,
    n_per_command: int,
    interp_steps:  int,
    device:        str,
    simulation_app,
):
    """
    Generate synthetic buffer transitions using POSITIONS dict.
 
    For each command with a known target position:
    1. Sample a random starting position near neutral
    2. Linearly interpolate to the target position
    3. Capture sim camera at each interpolation step
    4. Push transitions — success reward only at the terminal step
    """
    fs            = frame_stacks[0]
    total_pushed  = 0


    with tqdm(total=len(ALL_COMMANDS) * n_per_command,
          desc="Synthetic transitions",
          dynamic_ncols=True) as pbar:

        for cmd in ALL_COMMANDS:
            task = CMD_TO_TASK[cmd]

            if task not in POSITIONS:
                print(f"skipping {task} — not in POSITIONS")
                continue

            target = POSITIONS[task]

            for traj_idx in range(n_per_command):

                # start from home with small perturbation
                if task == "home":
                    start  = POSITIONS["reset"] 
                elif task == " neutral":
                    start  = POSITIONS["home"]
                else:
                    start  = POSITIONS["neutral"] 

                set_arm_joints(base_env, start, device)
                sim_step(base_env, simulation_app)
                update_frame_stack(base_env, [fs], reset_ids=[0])

                trajectory = np.linspace(start, target, interp_steps + 1)
                cam_cache = None
                for step_idx in range(interp_steps):
                    joint_pos    = trajectory[step_idx]
                    joint_next   = trajectory[step_idx + 1]
                    action_delta = joint_next - joint_pos
                    done         = (step_idx == interp_steps - 1)
                    reward       = 1.0 if done else 0.0
                    set_arm_joints(base_env, joint_pos, device)
                    sim_step(base_env, simulation_app)
                    transition, cam_cache = capture_transition(
                        base_env, fs, joint_pos, action_delta, reward,
                        joint_next, done, interp_steps - step_idx,
                        simulation_app, device, cam_state=cam_cache,
                    )
                    buffer.push(*transition)
                    total_pushed += 1

                pbar.set_postfix({
                    "cmd":    task,
                    "traj":   f"{traj_idx + 1}/{n_per_command}",
                    "buffer": len(buffer),
                    "pushed": total_pushed,
                }, refresh=True)
                pbar.update(1)


    print(f"synthetic generation done — buffer size: {len(buffer)}")
 
 

 
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
        )
    else:
        print("no lerobot_repo_id provided — skipping LeRobot conversion")
 
    # ── Source 2 — Synthetic POSITIONS ────────────────────────────────────────
    generate_synthetic_transitions(
        base_env, frame_stacks, buffer,
        n_per_command=args_cli.synthetic_per_cmd,
        interp_steps=args_cli.interp_steps,
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
