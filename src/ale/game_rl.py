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

def predict(ale_env_wrapped, c_frames, c_action, T):
    # get inner ALE for state save/restore
    inner = ale_env_wrapped
    while hasattr(inner, 'env'):
        inner = inner.env
    
    saved = inner.ale.cloneState()
    predict_frame = list(c_frames)

    for _ in range(T):
        obs, _, term, trunc, _ = ale_env_wrapped.step(c_action)
        if term or trunc:
            break
        # obs is [4, 84, 84] from FrameStackObservation — take last frame
        predict_frame.pop(0)
        predict_frame.append(obs[-1])

    inner.ale.restoreState(saved)
    return np.stack(predict_frame, axis=0)


def format_time(seconds):
    """
    Used to make time more readable
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def env_init(seed, N):
    """
    Used to create multiple envs and preprocess the data
    """
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
    """
    Runs the RL agent in training mode
    """

    #inital the environemt
    env = gym.vector.SyncVectorEnv([env_init(42, i) for i in range(args.environment)])
    brain = Brain(args.learning_rate,args.warmup, args.batch, args.gamma, args.tau, args.eps_end, args.eps_start, args.eps_decay, args.capacity)
    steps = 0
    episode_time = []

    #create or call a certain checkpoint
    if os.path.exists(f"runs/RL Agent-{args.job_name}/Checkpoints/brain{args.checkpoint}"):
        steps, start = brain.load_checkpoint()
        print("loading... brain", args.checkpoint )
    else:
        if not os.path.exists(f"runs/RL Agent-{args.job_name}/"):
            os.mkdir(f"runs/RL Agent-{args.job_name}/")
            os.mkdir(f"runs/RL Agent-{args.job_name}/Checkpoints/")
        steps, start = 0, 0 
    arg_name = f"RL Agent-{vars(args).pop("job_name")}"

    #starting logining in wandb
    if args.wandb:
        wandb.init(
            project="RL for Games",
            name=arg_name,
            config= args
        )
    #The main part 
    try:
        episode = start
        obs, _ = env.reset()
        state = obs

        #initalizing tracking metrics for the run
        total_reward = np.zeros(args.environment) #the total collected reward
        ep = [time.time()] * args.environment #the time for the episode 
        goal_reward = np.zeros(args.environment) #the reward taken directly from the ALE 
        tracking_reward = np.zeros(args.environment) #the reward calculated by following the ball direction
        prev_paddle_y = [None] * args.environment #the location of the paddle
        action_lap = np.zeros(args.environment) #simulate the action time of the robot arm
        actions = [{"all": 0, "up" : 0, "down": 0, "neutral": 0} for _ in range(args.environment)] #the amount of actions done
        prev_action = np.zeros(args.environment, dtype=np.int64) #the action prior to ensure no repeated actions
        
        with tqdm(total= args.episode, initial=start, desc="Training", unit="ep") as pbar:
            #runing as long as the episode count
            while episode < (args.episode +1):
                #tracks the overall time of a step

                #checks which env passed the execution limit
                ready = action_lap >= args.action_delay

                #predicting the next action and tracking overall the action metrics
                if args.predict:
                    lookahead_state = np.stack([
                        predict(env.envs[i], state[i], prev_action[i], args.action_delay)
                        for i in range(args.environment)
                    ])
                    action = brain.predict_next_action(lookahead_state, steps, env)
                else:
                    action = brain.predict_next_action(state, steps, env)

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
                action_lap = np.where(ready, 1, action_lap + 1)

                #move the environemtn forward and extracting necesary info 
                obs, raw_reward, terminated, truncated, _ = env.step(current_actions)
                reward = np.zeros(args.environment)
                done = terminated | truncated
                next_state = obs

                for i in range(args.environment):
                    #calcuate the goal reward direct from the env
                    if raw_reward[i] != 0:
                        goal = np.sign(raw_reward[i])
                        reward[i] += goal
                        goal_reward[i] += goal

                    #in case full reward is passed calcaulte the ball + paddle postion and then calcualte the reward for them
                    if args.full_rewards:
                        new_ball_y, new_paddle_y = brain.ball_position(next_state[i][-1])
                        if new_ball_y is not None and new_paddle_y is not None and prev_paddle_y[i] is not None:
                            new_distance = abs(new_ball_y - new_paddle_y)
                            prev_distance = abs(new_ball_y - prev_paddle_y[i])

                            #if clipping is used then only inc/dec by 1 if not then calculate the reward with scaling
                            if new_distance < prev_distance:
                                reward[i] += config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)
                                tracking_reward[i] +=  config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)

                            elif new_distance > config.THRESHOLD * config.CROP and new_distance >= prev_distance:
                                reward[i] -=  config.DISTANCE_REWARD * config.PENALTIY_MOVE
                                tracking_reward[i] -= config.DISTANCE_REWARD * config.PENALTIY_MOVE

                        prev_paddle_y[i] = new_paddle_y

                #clipped the reward if stated 
                clipped = np.clip(reward, -1, 1) if args.clip_reward else reward
                
                #populate the buffer
                for i in range(args.environment):
                    brain.buffer.push(state[i], current_actions[i], clipped[i], next_state[i], float(done[i]), steps)

                #tracking info
                total_reward += clipped
                steps += args.environment
                
                #passing the training depending on the updates called
                for _ in range(args.updates):
                    loss, grad_norm = brain.train()

                #logging the steps info
                if args.wandb:
                    wandb.log({
                        "train/loss":          loss,
                        "train/grad_norm":     grad_norm,
                        "train/epsilon":       brain.eps,
                        "train/buffer_size":   len(brain.buffer),
                        "train/episode":         episode,
                        "train/learning_rate": brain.optimiser.param_groups[0]["lr"],
                    }, step=steps)

                #after the episode being done, and calculating the necessary metrics
                for i in np.where(done)[0]:
                    ep_time = time.time() - ep[i]
                    episode_time.append(ep_time)
                    eta = np.mean(episode_time[-100:]) * (args.episode - episode - 1)

                    #saving when necessary
                    if episode % args.mid_save == 0: 
                        brain.save_checkpoint(episode, steps,arg_name)
                    if episode % args.full_save == 0 and episode != 0: 
                        brain.save()

                    #updating the progress bar
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

                    #updating the episode level metrics
                    if args.wandb:
                        wandb.log({
                            "episode/total_reward": total_reward[i],
                            "episode/goal_reward": goal_reward[i],
                            "episode/tracking_reward": tracking_reward[i],
                            "episode/clipped_reward": clipped[i],
                            "episode/action_all": actions[i]["all"],
                            "episode/actions_up": actions[i]["up"],
                            "episode/actions_down": actions[i]["down"],
                            "episode/actions_neutral": actions[i]["neutral"],
                        }, step=steps)

                    #restarting
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


    except Exception as e:

        #helps in logging any crashs
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        print(f"[CRASH] Episode: {episode} | Steps: {steps}", flush=True)
        env.close()
        if args.wandb:
            wandb.finish(exit_code=1)
        raise

    except KeyboardInterrupt:
        print("\nclosing...")
        env.close()
        wandb.finish()

def eval(args):
    """
    Runs the RL agent in eval mode
    """
    brain = Brain()
    brain.picking()
    
    #start the model on eval mode
    brain.policy.eval()

    #depending on the flag passed either camera or in game observation is used
    print("using real camera") if args.camera else print("using in game info") 
    frame = Eyes() if args.camera else Frames()
    
    #initalising the env
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    state = frame.reset() if args.camera else frame.reset(obs)
    
    #rendering the info in the proper screen for the camera
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    total_reward = 0
    count = 0
    while True:
        #main loop
        try:
            #calcultnig the state
            action = brain.rollout(state)

            #moving the state forward and rendering it 
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            render = env.render()
            surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (1920, 1080))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(60)
            total_reward += reward
            state = frame.reset() if args.camera else frame.reset(obs)
            
            if done:
                obs, _ = env.reset(seed=42)
                print(f"{count+1}.Evaluating {brain.agent}, the score is : {total_reward}")
                count += 1 
                total_reward = 0
                if count > args.iterations:
                    print("Done testing ", brain.agent)
                    break
                
        except KeyboardInterrupt:
            print("\nclosing...")
            env.close()
            break



def main():

    #The argumaents provided in the code
    parser = argparse.ArgumentParser("Training DQN for the Robot Arm")
    parser.add_argument("-env", "--environment", help="The amount of environment to run in sync for training the RL", type=int, default=config.ENV)
    parser.add_argument("-jn", "--job_name", help="Project name shown in wandb", type=str, default=str(random()))
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
    parser.add_argument("-tr", "--training", help="Toggle between training or eval", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-cam", "--camera", help="Toggle between using a camera or not", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-clip", "--clip_reward", help="Reward Clipping", default=True, action=argparse.BooleanOptionalAction)    
    parser.add_argument("-ad", "--action_delay", help="Number of env steps to hold an action before selecting the next one (simulates robot arm actuation delay)", type=int, default=0)
    parser.add_argument("-w",   "--wandb",         default=True,
                    action=argparse.BooleanOptionalAction)
    parser.add_argument("-p",   "--predict",         default=True,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.training:
        training(args)
    else:
        eval(args)


if __name__ == "__main__":
    main()