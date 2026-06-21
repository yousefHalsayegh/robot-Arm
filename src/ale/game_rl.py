"""
the ALE part training the robot
"""

import ale_py
import gymnasium as gym
import os
import numpy as np
import config
import time
from brain import Brain
from eyes import Eyes, Frames
import argparse
import wandb
from tqdm import tqdm
from random import random
import pygame

gym.register_envs(ale_py)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def env_init(seed, N):
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=1)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed + N)
        return env
    return _init

def training(args):
    
    if args.human_speed:
        slow = config.SLOW
    else:
        slow = 0

    
    env = gym.vector.SyncVectorEnv([env_init(42, i) for i in range(args.environment)])
    brain = Brain(args.learning_rate,args.warmup, args.batch, args.gamma, args.tau, args.eps_end, args.eps_start, args.eps_decay, args.capacity)
    steps = 0
    episode_time = []

    if os.path.exists(f"{args.job_name}/brain{args.checkpoint}"):
        steps, start = brain.load_checkpoint()
        print("loading... brain", args.checkpoint )
    else:
        if not os.path.exists(f"{args.job_name}/"):
            os.mkdir(f"{args.job_name}/Checkpoints/")
        steps, start = 0, 0 
    arg_name = vars(args).pop("job_name")

    wandb.init(
        project="RL for Games",
        name=arg_name,
        config= args
    )
    try:
        episode = start
        obs, _ = env.reset()
        state = obs

        total_reward = np.zeros(args.environment)
        ep = [time.time()] * args.environment
        goal_reward = np.zeros(args.environment)
        tracking_reward = np.zeros(args.environment)
        prev_paddle_y = [None] * args.environment
        action_lap = np.zeros(args.environment)
        actions = [{"all": 0, "up" : 0, "down": 0, "neutral": 0} for _ in range(args.environment)]
        prev_action = np.zeros(args.environment, dtype=np.int64)

        with tqdm(total= args.episode, initial=start, desc="Training", unit="ep") as pbar:
            while episode < (args.episode +1):
                lap = time.time()

                ready = np.ones(args.environment, dtype=bool) if not args.human_speed else action_lap >= args.execution

                action = brain.predict_next_action(state,steps, env)
                current_actions = np.where(ready, action, prev_action)
                for i in range(args.environment):
                    if ready[i] and prev_action[i] != current_actions[i]:
                        if current_actions[i] == 2:
                            actions[i]['up'] += 1
                        elif current_actions[i] == 3:
                            actions[i]['down'] += 1
                        else:
                            actions[i]['neutral'] += 1
                        actions[i]["all"] += 1
                prev_action = current_actions.copy()


                obs, raw_reward, terminated, truncated, _ = env.step(current_actions)
                reward = np.zeros(args.environment)
                done = terminated | truncated
                next_state = obs

                for i in range(args.environment):
                    
                    if raw_reward[i] != 0:
                        goal = np.sign(raw_reward[i])
                        reward[i] += goal
                        goal_reward[i] += goal


                    if args.full_rewards:
                        new_ball_y, new_paddle_y = brain.ball_position(next_state[i][-1])
                        if new_ball_y is not None and new_paddle_y is not None and prev_paddle_y[i] is not None:
                            new_distance = abs(new_ball_y - new_paddle_y)
                            prev_distance = abs(new_ball_y - prev_paddle_y[i])

                            if new_distance < prev_distance:
                                reward[i] += 1 if args.clip_reward else config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)
                                tracking_reward[i] += 1 if args.clip_reward else config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)

                            elif new_distance > config.THRESHOLD * config.CROP and new_distance >= prev_distance:
                                reward[i] -= 1 if args.clip_reward else config.DISTANCE_REWARD * config.PENALTIY_MOVE
                                tracking_reward[i] -= 1 if args.clip_reward else config.DISTANCE_REWARD * config.PENALTIY_MOVE

                        prev_paddle_y[i] = new_paddle_y

                clipped = np.clip(reward, -1, 1) if args.clip_reward else reward
                for i in range(args.environment):
                    brain.buffer.push(state[i], current_actions[i], clipped[i], next_state[i], float(done[i]))
                total_reward += clipped
                steps += args.environment
                
                for _ in range(args.updates):
                    loss, grad_norm = brain.train()


                wandb.log({
                    "train/loss":          loss,
                    "train/grad_norm":     grad_norm,
                    "train/epsilon":       brain.eps,
                    "train/buffer_size":   len(brain.buffer),
                    "train/steps":         steps,
                    "train/learning_rate": brain.optimiser.param_groups[0]["lr"],
                }, step=steps)

                for i in np.where(done)[0]:
                    ep_time = time.time() - ep[i]
                    episode_time.append(ep_time)
                    eta = np.mean(episode_time[-100:]) * (args.episode - episode - 1)
                    if episode % args.mid_save == 0 and episode != 0: 
                        brain.save_checkpoint(episode, steps,arg_name)
                    if episode % args.full_save == 0 and episode != 0: 
                        brain.save()

                    pbar.set_postfix({
                        "env": i,
                        "ep": episode,
                        "loss": f"{loss:.4f}",
                        "reward": f"{total_reward[i]:.1f}",
                        "goal": f"{goal_reward[i]:.0f}",
                        "track": f"{tracking_reward[i]:.1f}",
                        "eps": f"{brain.eps:.3f}",
                        "eta": format_time(eta),

                    })

                    pbar.update(1)

                    wandb.log({
                        "episode/total_reward": total_reward[i],
                        "episode/goal_reward": goal_reward[i],
                        "episode/tracking_reward": tracking_reward[i],
                        "episode/clipped_reward": clipped[i],
                        "episode/action_all": actions[i]["all"],
                        "episode/actions_up": actions[i]["up"],
                        "episode/actions_down": actions[i]["down"],
                        "episode/actions_neutral": actions[i]["neutral"],
                        "episode/episode": episode,
                    }, step=steps)

                    total_reward[i] = 0
                    tracking_reward[i] = 0
                    goal_reward[i] = 0
                    prev_paddle_y[i] = None
                    prev_action[i] = 0
                    action_lap[i] = 0
                    actions[i] = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
                    ep[i] = time.time()
                    episode += 1
                state = next_state

                overall = max(0, slow- (time.time() - lap))
                time.sleep(overall)
                action_lap += overall

                action_lap[ready] = 0.0

    except Exception as e:
    
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        print(f"[CRASH] Episode: {episode} | Steps: {steps}", flush=True)
        env.close()
        wandb.finish(exit_code=1)
        raise

    except KeyboardInterrupt:
        print("\nclosing...")
        env.close()
        wandb.finish()

def eval(check):
    options = []
    for i in os.listdir():
        if os.path.exists(f"{i}/brain4900.pth"):
            options.append(f"{i}/brain4900.pth")

    print("Pick from the list which Agent you would like to evalute:")
    for i in range(len(options)):
        print(f"{i+1}.{options[i].split("/")[0]}")
    brain = Brain()

    while True:
        try:
            choice = int(input()) - 1

            if choice > len(options) or choice < 0:
                print("Your option doesn't exist in the list, please pick something from the list")
                continue
            print("loading in ", options[choice])
            brain.load_checkpoint(options[choice])
            break

        except ValueError:
            print("Please enter a number")
            continue
    
    brain.policy.eval()
    print("using real camera") if check else print("using in game info") 
    frame = Eyes() if check else Frames()
    
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    state = frame.reset() if check else frame.reset(obs)
    
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    while True:
        try:
            action = brain.rollout(state)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            render = env.render()
            surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (1920, 1080))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(60)

            state = frame.reset() if check else frame.reset(obs)
            
            if done:
                obs, _ = env.reset(seed=42)
                check = not check
                print("switching to camera setting")
                frame = Eyes() if check else Frames()
        except KeyboardInterrupt:
            print("\nclosing...")
            env.close()
            break



def main():

    parser = argparse.ArgumentParser("Training DQN for the Robot Arm")
    parser.add_argument("-env", "--environment", help="The amount of environment to run in sync for training the RL", type=int, default=config.ENV)
    parser.add_argument("-jn", "--job_name", help="Project name shown in wandb", type=str, default=str(random()))
    parser.add_argument("-ex", "--execution", help="The speed of the robot arm", type=float, default=config.EXECUTION)
    parser.add_argument("-thld", "--threshold", help="The threshold distance between the paddle and the middle of the screen", type=float, default=config.THRESHOLD)
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
    parser.add_argument("-fr", "--full_rewards", help="This works with 3 rewards, rather than only goal", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-hs", "--human_speed", help="A checkpoint for the RL", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-tr", "--training", help="Toggle between training or eval", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-cam", "--camera", help="Toggle between using a camera or not", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-clip", "--clip_reward", help="Reward Clipping", default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    if args.training:
        training(args)
    else:
        eval(args.camera)


if __name__ == "__main__":
    main()