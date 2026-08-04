"""Launch Isaac Sim first."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Low-Level Joystick Controller Training")
parser.add_argument("--task",           type=str,  default=None)
parser.add_argument("--num_envs",       type=int,  default=8)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("-jn",  "--job_name",      type=str,   default="low_level")
parser.add_argument("-ep",  "--episodes",      type=int,   default=10000)
parser.add_argument("-fs",  "--full_save",     type=int,   default=500)
parser.add_argument("-md",  "--mid_save",      type=int,   default=100)
parser.add_argument("-chk", "--checkpoint",    type=str,   default="")
parser.add_argument("-w",   "--wandb",         default=True,
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--cam_embedding",   type=int, default=256)
parser.add_argument("--joint_embedding", type=int, default=64)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is up."""

import os
import time
import numpy as np
import torch
import wandb
from collections import deque
from tqdm import tqdm
import pickle
import gymnasium as gym
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from sim.scripts.fill_buffer import generate
from sim.utils.robo_brain import Brain

from sim.tasks.joystick.mdp.observations import (
    Frames, update_frame_stack
)
from sim.tasks.joystick.play_env_cfg import (
    ALL_COMMANDS, CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT,
    UPPER_THRESHOLD, LOWER_THRESHOLD, WINDOW_SIZE, STAGE_EPISODE_LENGTHS,
)
from sim.tasks.joystick.mdp.rewards import joystick_registered, ACT_TO_ZONE

CMD_NAMES = {
    CMD_NEUTRAL: "neutral", CMD_UP: "up",
    CMD_DOWN: "down", CMD_LEFT: "left", CMD_RIGHT: "right",
}


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def training(args, env, simulation_app):
    N        = args.num_envs
    base_env = env.unwrapped
    device   = str(base_env.device)

    # ── brain ─────────────────────────────────────────────────────────────────
    brain = Brain(
        ce   = args.cam_embedding,
        je = args.joint_embedding
    )

    steps, start_ep = 0, 0
    ckpt_dir  = f"runs/LowLevel-{args.job_name}/Checkpoints"
    ckpt_path = f"{ckpt_dir}/{args.checkpoint}.pth"
    if args.checkpoint and os.path.exists(ckpt_path):
        steps, start_ep = brain.load_checkpoint(ckpt_path)
        print(f"loaded: {ckpt_path}")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        print("no checkpoint, starting fresh")

    

    if args.wandb:
        wandb.init(
            project="RL for Games",
            name=f"LowLevel-{args.job_name}",
            config={k: v for k, v in vars(args).items()
                    if k not in {"job_name"}},
        )

    # ── frame stacks — one per env ────────────────────────────────────────────
    frame_stacks = [Frames(n=4) for _ in range(N)]

    prefill_path = "buffer_prefill.pkl"
    if os.path.exists(prefill_path):
        with open(prefill_path, "rb") as f:
            brain.buffer = pickle.load(f)
        print(f"loaded pre-filled buffer: {len(brain.buffer)} transitions")
    else:
        generate(env, simulation_app, brain.buffer, frame_stacks)
        with open(args_cli.output_path, "wb") as f:
                pickle.dump(brain.buffer, f)

    # ── curriculum success buffer ─────────────────────────────────────────────
    # passed to curriculum term so Isaac Lab can compute min success rate
    command_success_buf = {c: deque(maxlen=WINDOW_SIZE) for c in ALL_COMMANDS}
    
    current_stage = 0

    # ── per-env tracking ──────────────────────────────────────────────────────
    episode_steps   = np.zeros(N, dtype=int)
    episode_start_t = [time.time()] * N
    move_start_t    = [time.time()] * N
    failsafe_count  = np.zeros(N, dtype=int)
    episode         = start_ep
    episode_time    = []

    # decision state storage for n-step buffer push
    cam_decision   = None
    joint_decision = None

    critic_loss, actor_loss = 0.0, 0.0

    # ── initial reset ─────────────────────────────────────────────────────────
    obs, _ = env.reset()

    update_frame_stack(base_env, frame_stacks, reset_ids=list(range(N)))

    cam_states   = np.stack([fs._get_state() for fs in frame_stacks])
    joint_states = base_env.scene["robot"].data.joint_pos.cpu().numpy()

    cam_decision   = cam_states.copy()
    joint_decision = joint_states.copy()

    try:
        with tqdm(total=args.episodes, initial=start_ep,
                  desc="LowLevel Training", unit="ep") as pbar:
            while episode < args.episodes:

                # ── read current commands from Isaac Lab command manager ───────
                commands = base_env.command_manager.get_command(
                    "joystick_cmd"
                ).cpu().numpy()   # [N] int

                # ── action selection via ManipulationBrain ────────────────────
                actions = np.stack([
                    brain.predict_next_action(cam_states[i], joint_states[i], steps)
                    for i in range(N)
                ]) 

                # ── step Isaac Lab env ────────────────────────────────────────
                # action manager applies joint targets to SO101 via
                # JoystickActionTerm — no manual robot_sim.act() needed
                current_joints = base_env.scene["robot"].data.joint_pos.cpu().numpy()  # [N, 6]
                target_joints  = current_joints + actions                               # [N, 6]
                
                obs, rewards, terminated, truncated, info = env.step(
                    torch.tensor(target_joints, dtype=torch.float32, device=device)
                )
                dones = terminated | truncated

                # ── update frame stacks ───────────────────────────────────────
                reset_ids = torch.where(dones)[0].cpu().tolist()
                update_frame_stack(base_env, frame_stacks,
                                    reset_ids=reset_ids if reset_ids else None)

                cam_next   = np.stack([fs._get_state() for fs in frame_stacks])
                joint_next = base_env.scene["robot"].data.joint_pos.cpu().numpy()

                # ── per-env processing ────────────────────────────────────────
                for i in range(N):
                    episode_steps[i] += 1
                    done_i = bool(dones[i].item())

                    # read reward components from Isaac Lab reward manager
                    # rewards tensor is the combined reward from RewardsCfg
                    # individual components available via info if needed
                    combined_reward = brain.normalise_reward(
                        float(info.get("reward/sparse_success",
                              torch.zeros(N))[i]),
                        float(info.get("reward/step_penalty",
                              torch.zeros(N))[i]),
                        float(info.get("reward/axis_bonus",
                              torch.zeros(N))[i]),
                    )

                    if done_i or episode_steps[i] >= int(
                        base_env.cfg.episode_length_s * 100
                    ):
                        # check if success or timeout
                        task       = ACT_TO_ZONE[int(commands[i])]
                        registered = joystick_registered(base_env.scene["object"], i, task)

                        if not registered:
                            failsafe_count[i] += 1

                        # push to buffer
                        brain.buffer.push(
                            cam_decision[i],
                            joint_decision[i],
                            actions[i].copy(),
                            combined_reward,
                            cam_next[i],
                            joint_next[i],
                            float(registered),
                            int(episode_steps[i]),
                        )
                        steps += 1

                        # train
                        critic_loss, actor_loss = brain.train()

                        # movement time
                        move_time = time.time() - move_start_t[i]
                        ep_time   = time.time() - episode_start_t[i]
                        episode_time.append(ep_time)
                        eta = np.mean(episode_time[-100:]) * \
                              (args.episodes - episode - 1)

                        # update curriculum success buffer
                        cmd_i = int(commands[i])
                        command_success_buf[cmd_i].append(registered)

                        # check curriculum stage change
                        min_rate = min(
                            sum(command_success_buf[c]) /
                            max(len(command_success_buf[c]), 1)
                            for c in ALL_COMMANDS
                        )
                        if min_rate >= UPPER_THRESHOLD and current_stage < 2:
                            current_stage += 1
                            base_env.cfg.episode_length_s = \
                                STAGE_EPISODE_LENGTHS[current_stage]
                            # update event term randomisation range
                            _update_curriculum_stage(base_env, current_stage)
                            print(f"curriculum advanced to stage {current_stage}",
                                  flush=True)
                        elif (1.0 - min_rate) >= LOWER_THRESHOLD \
                                and current_stage > 0:
                            current_stage -= 1
                            base_env.cfg.episode_length_s = \
                                STAGE_EPISODE_LENGTHS[current_stage]
                            _update_curriculum_stage(base_env, current_stage)
                            print(f"curriculum regressed to stage {current_stage}",
                                  flush=True)

                        # wandb
                        if args.wandb:
                            per_cmd_rates = {
                                f"success_rate/{CMD_NAMES[c]}":
                                    sum(command_success_buf[c]) /
                                    max(len(command_success_buf[c]), 1)
                                for c in ALL_COMMANDS
                            }
                            wandb.log({
                                "train/critic_loss": critic_loss,
                                "train/actor_loss":  actor_loss,
                                "train/alpha":       brain.alpha.item(),
                                "train/buffer_size":     len(brain.buffer),
                                "train/steps":           steps,
                                "episode/success":       float(registered),
                                "episode/steps_taken":   episode_steps[i],
                                "episode/episode_time":  ep_time,
                                "episode/move_time":     move_time,
                                "episode/command":       CMD_NAMES.get(
                                                             cmd_i, str(cmd_i)),
                                "episode/failsafe":      failsafe_count[i],
                                "episode/episode":       episode,
                                "curriculum/stage":      current_stage,
                                "curriculum/min_rate":   min_rate,
                                "curriculum/ep_length":
                                    base_env.cfg.episode_length_s,
                                **per_cmd_rates,
                            }, step=steps)

                        pbar.set_postfix({
                            "ep":      episode,
                            "cmd":     CMD_NAMES.get(cmd_i, str(cmd_i)),
                            "success": registered,
                            "steps":   episode_steps[i],
                            "stage":   current_stage,
                            "eta":     format_time(eta),
                        })
                        pbar.update(1)

                        print(
                            f"\n[EP {episode}] env={i} | "
                            f"cmd={CMD_NAMES.get(cmd_i, str(cmd_i))} | "
                            f"success={registered} | "
                            f"steps={episode_steps[i]} | "
                            f"stage={current_stage} | "
                            f"move_time={move_time:.2f}s | "
                            f"min_rate={min_rate:.2f} | "
                            f"eta={format_time(eta)}",
                            flush=True,
                        )

                        if episode % args.mid_save == 0 and episode != 0:
                            brain.save_checkpoint(
                                episode, steps,
                                f"LowLevel-{args.job_name}/Checkpoints"
                            )
                        if episode % args.full_save == 0 and episode != 0:
                            brain.save_checkpoint(
                                episode, steps,
                                f"LowLevel-{args.job_name}/Full"
                            )

                        # reset per-env trackers
                        episode_steps[i]   = 0
                        failsafe_count[i]  = 0
                        episode_start_t[i] = time.time()
                        move_start_t[i]    = time.time()
                        episode           += 1

                        # update decision state reference
                        cam_decision[i]   = cam_next[i].copy()
                        joint_decision[i] = joint_next[i].copy()

                # update states for next step
                cam_states   = cam_next
                joint_states = joint_next

    except KeyboardInterrupt:
        print("\nclosing")
        if args.wandb:
            wandb.finish()

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
                title=f"LowLevel crashed — {type(e).__name__}",
                text=f"Episode {episode} | Steps {steps}\n\n{crash}",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=1)
        raise


def _update_curriculum_stage(base_env, stage: int):
    """Update event term randomisation range when stage changes."""
    stage_params = {
        0: {"pos_range": 0.00, "rot_range": 0.00},
        1: {"pos_range": 0.05, "rot_range": 0.05},
        2: {"pos_range": 0.15, "rot_range": 0.15},
    }
    params = stage_params.get(stage, stage_params[0])
    base_env.cfg.events.reset_controller.params.update(params)


def main():
    import sim.tasks.joystick.play_env_cfg as _cfg_module

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