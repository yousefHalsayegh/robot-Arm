"""
game_sim.py

Isaac Sim equivalent of game_rob.py.

Mirrors game_rob.py exactly:
- Action gating: new action only when arm finishes previous one
- Same fail_safe counter per env
- Same Brain, Frames, Eyes unchanged
- ALE runs with render_mode="human" alongside Isaac Sim
- Isaac Sim handles arm physics only — all articulation calls batched
- Optional LeRobot dataset recording (-rec flag)
- Optional per-episode colour randomisation (-cr flag)

Run:
    python game_sim.py --task <your_task> --num_envs 4 --job_name sim_run_1
"""

"""Launch Isaac Sim first."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Isaac Sim DQN — Robot Arm Pong")
parser.add_argument("--task",           type=str,   default=None)
parser.add_argument("--num_envs",       type=int,   default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("-jn",  "--job_name",      type=str,   default="sim_run")
parser.add_argument("-ep",  "--episode",       type=int,   default=5000)
parser.add_argument("-fs",  "--full_save",     type=int,   default=500)
parser.add_argument("-md",  "--mid_save",      type=int,   default=100)
parser.add_argument("-lr",  "--learning_rate", type=float, default=0.0001)
parser.add_argument("-wp",  "--warmup",        type=int,   default=10000)
parser.add_argument("-b",   "--batch",         type=int,   default=128)
parser.add_argument("-tau", "--tau",           type=float, default=0.005)
parser.add_argument("-ee",  "--eps_end",       type=float, default=0.05)
parser.add_argument("-es",  "--eps_start",     type=float, default=1.0)
parser.add_argument("-ed",  "--eps_decay",     type=float, default=500000)
parser.add_argument("-g",   "--gamma",         type=float, default=0.99)
parser.add_argument("-c",   "--capacity",      type=float, default=100000)
parser.add_argument("-chk", "--checkpoint",    type=str,   default="")
parser.add_argument("-cam", "--camera",        default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument("-rec", "--record",        default=False,
                    action=argparse.BooleanOptionalAction,
                    help="record LeRobot dataset alongside training")
parser.add_argument("-cr",  "--colour_rand",   default=True,
                    action=argparse.BooleanOptionalAction,
                    help="randomise asset colours each episode")
parser.add_argument("--arm_prim",   type=str, default="Robot")
parser.add_argument("--stick_prim", type=str, default="arcade_stick")
parser.add_argument("--table_prim", type=str, default=None,
                    help="table prim name or full path, e.g. Table or /World/Table")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is up."""

import gymnasium as gym
import numpy as np
import torch
import os
import time
import wandb
from tqdm import tqdm
from collections import deque

import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.utils.stage as stage_utils

from ale.brain import Brain
from ale.eyes import Eyes, Frames
from sim.utils.robot import (
    RobotSim, POSITIONS, ARRIVAL_THRESHOLD,
    send_targets, batch_move_arms,
)
from sim.utils.record import LeRobotRecorder
from sim.utils.colours import set_sim_colours, randomise_asset_colours

import ale_py
gym.register_envs(ale_py)


def format_time(seconds):
    d = int(seconds // 86400)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{d:02d}:{h:02d}:{m:02d}:{s:02d}"


def make_ale_envs(N: int, seed: int = 42):
    """Create N independent ALE Pong envs with human rendering."""
    return [
        gym.make("ALE/Pong-v5", frameskip=1, render_mode="human")
        for _ in range(N)
    ]


def training(args, env, simulation_app):
    N        = args.num_envs
    base_env = env.unwrapped
    device   = str(base_env.device)
    stage    = stage_utils.get_current_stage()

    # ── brain ─────────────────────────────────────────────────────
    brain = Brain(
        args.learning_rate, args.warmup, args.batch,
        args.gamma, args.tau, args.eps_end,
        args.eps_start, args.eps_decay, args.capacity,
    )

    steps, start = 0, 0
    ckpt_dir  = f"Arm-{args.job_name}/Checkpoints"
    ckpt_path = f"{ckpt_dir}/brain{args.checkpoint}.pth"
    if args.checkpoint and os.path.exists(ckpt_path):
        steps, start = brain.load_checkpoint(ckpt_path)
        print(f"loaded: {ckpt_path}")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        print("no checkpoint, starting fresh")

    wandb.init(
        project="RL for Games",
        name=f"Arm-Sim-{args.job_name}",
        config={k: v for k, v in vars(args).items()
                if k not in {"job_name"}},
    )

    # ── observation source ────────────────────────────────────────
    print("using real camera") if args.camera else print("using in game info")
    if args.camera:
        eyes   = [Eyes() for _ in range(N)]
        frames = [Frames() for _ in range(N)]
        switch = [True] * N
    else:
        frames = [Frames() for _ in range(N)]
        switch = [False] * N

    # ── ALE envs — human render, no screen prim needed ───────────
    ale_envs = make_ale_envs(N)
    ale_obs  = [ale_envs[i].reset(seed=42 + i)[0] for i in range(N)]
    states   = [
        eyes[i].reset() if switch[i] else frames[i].reset(ale_obs[i])
        for i in range(N)
    ]

    # ── articulation ──────────────────────────────────────────────
    so101  = base_env.scene["robot"]
    robots = [RobotSim(env_index=i) for i in range(N)]

    def sim_step():
        base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())
        simulation_app.update()

    # ── apply physical colours ────────────────────────────────────
    # replace RGB values with output from: python colour_tools.py --sample
    set_sim_colours(
        stage      = stage,
        num_envs   = N,
        arm_rgb    = (0.85, 0.75, 0.60),
        stick_rgb  = (0.10, 0.10, 0.12),
        table_rgb  = (0.72, 0.58, 0.42),
        arm_prim   = args.arm_prim,
        stick_prim = args.stick_prim,
        table_prim = args.table_prim,
    )

    # ── recorder ─────────────────────────────────────────────────
    recorder    = LeRobotRecorder(f"lerobot_dataset/{args.job_name}") \
                  if args.record else None
    frame_idxs  = np.zeros(N, dtype=int)
    ep_start_ts = [time.time()] * N

    # ── move all arms to home ─────────────────────────────────────
    batch_move_arms(so101, robots, sim_step, "home", device)

    # ── per-env tracking ──────────────────────────────────────────
    total_rewards  = np.zeros(N)
    episode        = start
    ep_times       = [time.time()] * N
    episode_time   = []
    actions        = [{"all": 0, "up": 0, "down": 0, "neutral": 0}
                      for _ in range(N)]
    current_acts   = np.zeros(N, dtype=np.int64)
    fail_safes     = np.zeros(N, dtype=int)
    loss, grad_norm = 0.0, 0.0

    try:
        with tqdm(total=args.episode, initial=start,
                  desc="Training", unit="ep") as pbar:
            while episode < (args.episode + 1):

                # ── action gating per env — mirrors game_rob.py ───
                for i in range(N):
                    robot    = robots[i]
                    arm_done = robot.finished(so101)

                    if (robot.prev == robot.action
                            and robot.action == current_acts[i]) \
                       or (fail_safes[i] >= 10 and not arm_done):

                        act = brain.predict_next_action(
                            states[i], steps, ale_envs[i]
                        )

                        if act in (2, 4):
                            act          = 2
                            robot.task   = "up"
                            actions[i]["up"] += 1
                        elif act in (3, 5):
                            act            = 3
                            robot.task     = "down"
                            actions[i]["down"] += 1
                        else:
                            act               = 0
                            robot.task        = "neutral"
                            actions[i]["neutral"] += 1

                        actions[i]["all"] += 1
                        current_acts[i]    = act
                        robot.action       = act
                        robot.prev         = act
                        fail_safes[i]      = 0

                # ── batched arm command + sim step ────────────────
                send_targets(so101, robots, device)
                sim_step()

                # ── step each ALE env with robot's current action ─
                next_states = []
                for i in range(N):
                    o, rew, term, trunc, _ = ale_envs[i].step(
                        int(robots[i].action)
                    )
                    done_i = term or trunc

                    ns = eyes[i].step() if switch[i] \
                         else frames[i].step(o)
                    next_states.append(ns)

                    # record step if enabled
                    if recorder is not None:
                        side_frame = base_env.scene["side"]\
                            .data.output["rgb"][i].cpu().numpy()\
                            .astype(np.uint8)
                        recorder.record_step(
                            joint_pos    = robots[i].get_joint_pos(so101),
                            joint_target = POSITIONS[robots[i].task],
                            side_frame   = side_frame,
                            frame_index  = int(frame_idxs[i]),
                            timestamp    = time.time() - ep_start_ts[i],
                            done         = done_i,
                        )
                        frame_idxs[i] += 1

                    # reward + buffer
                    clipped = np.clip(np.sign(rew), -1, 1)
                    brain.buffer.push(
                        states[i], current_acts[i], clipped,
                        ns, float(done_i)
                    )
                    total_rewards[i] += clipped
                    fail_safes[i]    += 1

                    # train
                    result = brain.train()
                    if result is not None:
                        loss, grad_norm = result

                    steps += 1

                    # ── episode end ───────────────────────────────
                    if done_i:
                        ep_time = time.time() - ep_times[i]
                        episode_time.append(ep_time)
                        eta = np.mean(episode_time[-100:]) * \
                              (args.episode - episode - 1)

                        if episode % args.mid_save == 0 and episode != 0:
                            brain.save_checkpoint(
                                episode, steps, f"Arm-{args.job_name}"
                            )
                        if episode % args.full_save == 0 and episode != 0:
                            brain.save()

                        pbar.set_postfix({
                            "env":    i,
                            "ep":     episode,
                            "loss":   f"{loss:.4f}",
                            "reward": f"{total_rewards[i]:.1f}",
                            "eps":    f"{brain.eps:.3f}",
                            "eta":    format_time(eta),
                        })
                        pbar.update(1)

                        print(
                            f"\n[EP {episode}] env={i} | "
                            f"reward={total_rewards[i]:.1f} | "
                            f"loss={loss:.4f} | "
                            f"eps={brain.eps:.3f} | "
                            f"eta={format_time(eta)}",
                            flush=True,
                        )

                        wandb.log({
                            "train/loss":          loss,
                            "train/grad_norm":     grad_norm,
                            "train/epsilon":       brain.eps,
                            "train/buffer_size":   len(brain.buffer),
                            "train/learning_rate": brain.optimiser.param_groups[0]["lr"],
                            "episode/total_reward":       total_rewards[i],
                            "episode/RL_action_all":      actions[i]["all"],
                            "episode/RL_actions_up":      actions[i]["up"],
                            "episode/RL_actions_down":    actions[i]["down"],
                            "episode/RL_actions_neutral": actions[i]["neutral"],
                            "episode/RB_action_all":      robots[i].actions["all"],
                            "episode/RB_actions_up":      robots[i].actions["up"],
                            "episode/RB_actions_down":    robots[i].actions["down"],
                            "episode/RB_actions_neutral": robots[i].actions["neutral"],
                        }, step=steps)

                        # flush recorder episode
                        if recorder is not None:
                            recorder.end_episode()
                            frame_idxs[i]  = 0
                            ep_start_ts[i] = time.time()

                        # reset per-env trackers
                        total_rewards[i]       = 0.0
                        robots[i].action       = 0
                        robots[i].prev         = 0
                        robots[i].actions      = {"all": 0, "up": 0,
                                                  "down": 0, "neutral": 0}
                        actions[i]             = {"all": 0, "up": 0,
                                                  "down": 0, "neutral": 0}
                        current_acts[i]        = 0
                        fail_safes[i]          = 0
                        ep_times[i]            = time.time()
                        episode               += 1

                        # reset ALE
                        ale_obs_i, _ = ale_envs[i].reset(seed=42)

                        # camera switch every 10 episodes — mirrors game_rob.py
                        if episode > 1 and episode % 10 == 0 and args.camera:
                            switch[i] = not switch[i]
                            print(f"env {i}: "
                                  + ("camera" if switch[i] else "in-game"))

                        states[i] = eyes[i].reset() if switch[i] \
                                    else frames[i].reset(ale_obs_i)

                        # colour randomisation
                        if args.colour_rand:
                            randomise_asset_colours(
                                stage,
                                env_index  = i,
                                arm_prim   = args.arm_prim,
                                stick_prim = args.stick_prim,
                                table_prim = args.table_prim,
                            )

                        # reset arm to neutral for this env only
                        # keep other envs running — only pause this one
                        robots[i].task = "neutral"

                    else:
                        states[i] = next_states[i]

                wandb.log({"train/steps": steps}, step=steps)

    except KeyboardInterrupt:
        print("\nclosing")
        if recorder is not None:
            recorder.save()
        for e in ale_envs:
            e.close()
        wandb.finish()

    except Exception as e:
        import traceback
        crash = traceback.format_exc()
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        print(crash, flush=True)
        if recorder is not None:
            recorder.save()
        wandb.log({
            "crash/error_type":    type(e).__name__,
            "crash/error_message": str(e),
            "crash/traceback":     crash,
            "crash/episode":       episode,
            "crash/steps":         steps,
        })
        wandb.alert(
            title=f"Sim crashed — {type(e).__name__}",
            text=f"Episode {episode} | Steps {steps}\n\n{crash}",
            level=wandb.AlertLevel.ERROR,
        )
        for e in ale_envs:
            e.close()
        wandb.finish(exit_code=1)
        raise


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())

    training(args_cli, env, simulation_app)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()