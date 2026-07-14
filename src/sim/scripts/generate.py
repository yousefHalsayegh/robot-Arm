

import argparse
from isaaclab.app import AppLauncher
import ale.config as config

parser = argparse.ArgumentParser("Isaac Sim DQN — Robot Arm Pong")
parser.add_argument("--task",type=str,default=None)
parser.add_argument("--num_envs",type=int,default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("-jn", "--job_name", help="Project name shown in wandb", type=str, default="Sim")
parser.add_argument("-ep", "--episode", help="The amount of episodes to train for in total", type=int, default=config.EPISODES)
parser.add_argument("-u", "--updates", help="Per episode how many times do we run the train method for the RL", type=int, default=config.UPDATES)
parser.add_argument("-fs", "--full_save", help="The episode to save the model", type=int, default=config.FULL_SAVE)
parser.add_argument("-md", "--mid_save", help="The episode to save the model, with the extra information", type=int, default=config.MID_SAVE)
parser.add_argument("-lr", "--learning_rate", help="The learning rate for the agent", type=float, default=config.LEARNING_RATE)
parser.add_argument("-wp", "--warmup", help="The steps needed before training start fully, to give room for the buffer", type=int, default=config.WARMUP)
parser.add_argument("-b", "--batch", help="The amount batches taken from the buffer", type=int, default=config.BATCH)
parser.add_argument("-tau", "--tau", help="Helps in the soft update of the policy and the target netwrok", type=float, default=config.TAU)
parser.add_argument("-ee", "--eps_end", help="The end point of epsilon", type=float, default=config.EPS_END)
parser.add_argument("-es", "--eps_start", help="The starting point of the epsilon for exploration", type=float, default=config.EPS_START)
parser.add_argument("-ed", "--eps_decay", help="The overall rate for the epsilon to decay", type=float, default=config.EPS_DECAY)
parser.add_argument("-g", "--gamma", help="This helps with the discounted rate of the reward", type=float, default=config.GAMMA)
parser.add_argument("-c", "--capacity", help="The replay buffer capacity", type=float, default=config.CAPACITY)
parser.add_argument("-chk", "--checkpoint", help="A checkpoint for the RL", type=str, default=config.CHECKPOINT)
parser.add_argument("-rec", "--record", default=False, action=argparse.BooleanOptionalAction, help="record LeRobot dataset alongside training")
parser.add_argument("-w", "--wandb", default=True, action=argparse.BooleanOptionalAction, help="record the info in wandb")
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
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.utils.stage as stage_utils
import cv2
from ale.brain import Brain
from sim.utils.robot_sim import (
    RobotSim, POSITIONS,
    send_targets, batch_move_arms,
    batch_move_arm,
)
from sim.utils.pong_display import PongDisplay

import ale_py
gym.register_envs(ale_py)

def predict(ale_env, c_frames, c_action,T ):
    save = ale_env.ale.cloneState()

    predict_frame = list(c_frames)

    for _ in range(T):
        ale_env.step(c_action)

        frame = ale_env.ale.getScreenGrayscale().astype(np.float32) / 255.0
        frame = cv2.resize(frame, (84, 84))

        predict_frame.pop(0)
        predict_frame.append(frame)

    ale_env.ale.restoreState(save)

    return np.stack(predict_frame, axis=0)


def format_time(seconds):
    d = int(seconds // 86400)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{d:02d}:{h:02d}:{m:02d}:{s:02d}"


def env_init(seed, rank):
    """Factory function for SyncVectorEnv."""
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True,
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed + rank)
        return env
    return _init

def joystick_zone(object_art, env_index):
    """Classify the joystick's CURRENT physical position into a zone,
    independent of what task it's being driven toward. This is what the
    game should actually see, moment to moment — same as a real controller."""
    tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())  # [PivotY, PivotX]
    axis_deg = tilt_deg[PIVOT_X_IDX]

    if axis_deg < -DEADZONE_DEG:
        return "up"
    elif axis_deg > DEADZONE_DEG:
        return "down"
    return "neutral"


ZONE_TO_ACT = {"up": 2, "down": 3, "neutral": 0}

DEADZONE_DEG = 6.5 

# joint order from your earlier print: ['PivotY', 'PivotX']
PIVOT_Y_IDX = 0
PIVOT_X_IDX = 1

def joystick_registered(object_art, env_index, task):
    tilt = object_art.data.joint_pos[env_index].cpu().numpy()  # [PivotY, PivotX], radians
    tilt_deg = np.rad2deg(tilt)
    if task == "neutral":
        return np.abs(tilt_deg).max() < DEADZONE_DEG

    axis_deg = tilt_deg[PIVOT_X_IDX]
    if task == "up":
        return axis_deg < -DEADZONE_DEG
    elif task == "down":
        return axis_deg > DEADZONE_DEG
    return False

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
    ckpt_dir  = f"Sim-{args.job_name}/Checkpoints"
    ckpt_path = f"{ckpt_dir}/brain{args.checkpoint}.pth"
    if args.checkpoint and os.path.exists(ckpt_path):
        steps, start = brain.load_checkpoint(ckpt_path)
        print(f"loaded: {ckpt_path}")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        print("no checkpoint, starting fresh")

    if args.wandb:
        wandb.init(
            project="RL for Games",
            name=f"Sim-{args.job_name}",
            config={k: v for k, v in vars(args).items()
                    if k not in {"job_name"}},
        )

    # ── ALE envs via SyncVectorEnv ────────────────────────────────
    # AtariPreprocessing + FrameStackObservation handle all preprocessing
    # obs shape out: [N, 4, 84, 84] float32 — used directly as state
    ale_envs = gym.vector.SyncVectorEnv(
        [env_init(42, i) for i in range(N)]
    )

    # obs: [N, 4, 84, 84] — states used directly, no Frames/Eyes needed
    obs, _ = ale_envs.reset()
    states  = obs.copy()   # [N, 4, 84, 84]

    # ── Pong display inside Isaac Sim ─────────────────────────────
    display = PongDisplay(num_envs=N)

    # ── articulation ──────────────────────────────────────────────
    so101  = base_env.scene["robot"]
    object_art = base_env.scene["object"]
    robots = [RobotSim(env_index=i) for i in range(N)]

    def sim_step():
        base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())
        simulation_app.update()

    # ── move all arms to home ─────────────────────────────────────
    batch_move_arms(so101, robots, sim_step, "reset", device)

    # ── per-env tracking ──────────────────────────────────────────
    total_rewards = np.zeros(N)
    episode       = start
    ep_times      = [time.time()] * N
    episode_time  = []
    actions       = [{"all": 0, "up": 0, "down": 0, "neutral": 0}
                     for _ in range(N)]
    current_acts  = np.zeros(N, dtype=np.int64)
    failsafe_Count  = np.zeros(N, dtype=np.int64)
    failsafe  = np.zeros(N, dtype=np.int64)
    loss, grad_norm = 0.0, 0.0
    decision_states = states
    pending_reward = np.zeros(N, dtype=np.int64)
    batch_move_arms(so101, robots, sim_step, "home", device)
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    try:
        with tqdm(total=args.episode, initial=start,
                  desc="Training", unit="ep") as pbar:
            while episode < (args.episode + 1):
                
                # ── action gating per env ─────────────────────────
                for i in range(N):
                    robot    = robots[i]
                    joystick_input = joystick_registered(object_art, i, robot.task)
                    timeout = failsafe[i] > 60
                    # if joystick_input:
                    #     print(f"env {i} registered at failsafe={failsafe[i]}", flush=True)
                    if timeout:
                        failsafe_Count[i] += 1
                    if joystick_input or timeout:

                        clipped_r = float(np.clip(pending_reward[i], -1, 1))
                        brain.buffer.push(
                            decision_states[i], int(current_acts[i]),
                            clipped_r, states[i], False, failsafe[i]
                        )
                        total_rewards[i] += clipped_r
                        steps            += 1
                        pending_reward[i] = 0

                        for _ in range(args.updates):
                            loss, grad_norm = brain.train()

                        predict(ale_envs[i].unwrapped, list(states[i]), int(current_acts[i]), 50)
                        act = brain.predict_next_action(states[i], steps, ale_envs)
                        if act in (2, 4):
                            robot.task       = "up"
                            actions[i]["up"] += 1
                        elif act in (3, 5):
                            robot.task         = "down"
                            actions[i]["down"] += 1
                        else:
                            robot.task            = "neutral"
                            actions[i]["neutral"] += 1
                        actions[i]["all"] += 1
                        failsafe[i] =0
                        decision_states[i] = states[i].copy()
                    current_acts[i] = ZONE_TO_ACT[joystick_zone(object_art, i)]
                    failsafe += 1

                # ── batched arm command + sim step ────────────────
                send_targets(so101, robots, device)
                sim_step()

                # ── step ALL ALE envs at once ─────────────────────
                # returns [N, 4, 84, 84] obs — used directly as next_states
                next_obs, rew_batch, term_batch, trunc_batch, _ = \
                    ale_envs.step(current_acts.copy())
                dones_batch = term_batch | trunc_batch

                # ── update Pong display ───────────────────────────
                for i in range(N):
                    try:
                        frame = ale_envs.envs[i].render()
                        display.update(i, frame)
                    except Exception:
                        pass
                simulation_app.update()

                # ── per-env processing ────────────────────────────
                for i in range(N):
                    done_i   = bool(dones_batch[i])
                    rew      = float(rew_batch[i])
                    clipped_r = float(np.clip(np.sign(rew), -1, 1))
                    pending_reward[i] += (brain.gamma ** failsafe[i]) * clipped_r
                    # ── episode end ───────────────────────────────
                    if done_i:
                        # reward + buffer
                        brain.buffer.push(
                            decision_states[i], int(current_acts[i]),
                            clipped_r, next_obs[i], True, failsafe[i]
                        )
                        total_rewards[i] += clipped_r
                        steps            += 1

                        # train
                        for _ in range(args.updates):
                            loss, grad_norm = brain.train()
                            
                        batch_move_arm(so101, robots[i], sim_step, "neutral", device)
                        ep_time = time.time() - ep_times[i]
                        episode_time.append(ep_time)
                        eta = np.mean(episode_time[-100:]) * \
                              (args.episode - episode - 1)

                        if episode % args.mid_save == 0 and episode != 0:
                            brain.save_checkpoint(
                                episode, steps, f"Sim-{args.job_name}"
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
                        if args.wandb:
                            wandb.log({
                                "train/loss":          loss,
                                "train/grad_norm":     grad_norm,
                                "train/epsilon":       brain.eps,
                                "train/buffer_size":   len(brain.buffer),
                                "train/learning_rate": brain.optimiser.param_groups[0]["lr"],
                                "train/episode": episode,
                                "episode/total_reward":       total_rewards[i],
                                "episode/pending_reward":       pending_reward[i],
                                "episode/RL_action_all":      actions[i]["all"],
                                "episode/RL_actions_up":      actions[i]["up"],
                                "episode/RL_actions_down":    actions[i]["down"],
                                "episode/RL_actions_neutral": actions[i]["neutral"],
                                "episode/flushed": failsafe_Count[i]
                            }, step=steps)


                        # reset per-env trackers
                        total_rewards[i]  = 0.0
                        robots[i].actions = {"all": 0, "up": 0,
                                             "down": 0, "neutral": 0}
                        actions[i]        = {"all": 0, "up": 0,
                                             "down": 0, "neutral": 0}
                        current_acts[i]   = 0
                        ep_times[i]       = time.time()
                        episode          += 1

                        # SyncVectorEnv auto-resets done envs
                        # next_obs[i] is already the fresh reset obs
                        states[i] = next_obs[i]
                        failsafe_Count[i] = 0
                        decision_states[i] = next_obs[i].copy()

                        robots[i].task = "neutral"

                    else:
                        states[i] = next_obs[i]


    except KeyboardInterrupt:
        print("\nclosing")
        ale_envs.close()
        if args.wandb:wandb.finish()

    except Exception as e:
        import traceback
        crash = traceback.format_exc()
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        print(crash, flush=True)
        if args.wandb:
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
            ale_envs.close()
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
    simulation_app.close()


if __name__ == "__main__":
    main()
    