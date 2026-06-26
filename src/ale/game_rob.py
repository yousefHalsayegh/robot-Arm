"""
the ALE part training the robot
"""

import ale_py
import gymnasium as gym
import threading
import os
import numpy as np
import json
import config
import time
from robot_noVla import Robot
from eyes import Eyes, Frames
from brain import Brain
import pygame
import argparse
import wandb
from tqdm import tqdm
from random import random

gym.register_envs(ale_py)
def format_time(seconds):
    """
    Used to make time more readable
    """
    d = int(seconds //86400)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{d:02d}:{h:02d}:{m:02d}:{s:02d}"

def training(args):
    robot = Robot()

    #in case there is already saved positons use them
    if os.path.exists("positions.json"):
        with open("positions.json", "r") as f:
            temp = json.load(f)
        robot.positions = {keys:values for keys, values in temp.items()}
    
    #start the robot and put it at home
    robot.start()
    robot.inital()

    #pick the agent to test on 
    brain = Brain(args.learning_rate,args.warmup, args.batch, args.gamma, args.tau, args.eps_end, args.eps_start, args.eps_decay, args.capacity)
    
    #create or call certain checkpoint
    if os.path.exists(f"Arm-{args.job_name}/Checkpoints/brain{args.checkpoint}.pth"):
        steps, start = brain.load_checkpoint(f"Arm-{args.job_name}/Checkpoints/brain{args.checkpoint}.pth")
        print("loading... brain", args.checkpoint )
    else:
        print("no check point")
        if not os.path.exists(f"Arm-{args.job_name}/"):
            os.mkdir(f"Arm-{args.job_name}/")
            os.mkdir(f"Arm-{args.job_name}/Checkpoints/")
        steps, start = 0, 0 
    arg_name = f"Arm-{vars(args).pop("job_name")}"

    #starting logining in wandb
    wandb.init(
        project="RL for Games",
        name=arg_name,
        config= args
    )

    print("using real camera") if args.camera else print("using in game info") 
    if args.camera:
        frame = Frames()
        switch = True
        eye = Eyes()
    else:
        switch = False
        frame = Frames()

    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    state = eye.reset() if switch else frame.reset(obs)

    #start threading for reading the input, move the arm and playing the game
    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()
    with open("positions.json", "w") as f:
        json.dump(robot.positions, f, indent=2)

    #render the screen on a specific display
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    total_reward = 0
    episode = start 
    ep = time.time()
    episode_time = []
    actions = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
    prev_action = 0
    action = 0
    fail_safe = 0
    try :
        with tqdm(total= args.episode, initial=start, desc="Training", unit="ep") as pbar:
            while episode < (args.episode +1):
                if (prev_action == robot.action and robot.action == action) or (fail_safe >= 10 and robot.action != action):
                        action = brain.predict_next_action(state, steps, env)
                        if action == 2 or action ==4:
                            action = 2
                            actions["up"] += 1
                            robot.task = "up"
                        elif action == 3 or action == 5:
                            action = 3
                            robot.task = "down"
                            actions["down"] += 1
                        elif action == 0 or action == 1:
                            action = 0
                            robot.task = "neutral"
                            actions["neutral"] += 1
                        else:
                            print('no clue')
                        actions["all"] += 1
                        prev_action = action
                        fail_safe= 0

                    #move the state forward
                obs, reward, terminated, truncated, _ = env.step(robot.action)
                done = terminated or truncated
                next_state = eye.step() if switch else frame.step(obs)

                #render the stateactions = [{"all": 0, "up" : 0, "down": 0, "neutral": 0}
                render = env.render()
                surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
                scaled = pygame.transform.scale(surf, (1920, 1080))
                screen.blit(scaled, (0, 0))
                pygame.display.flip()
                clock.tick(60)
                
                #reward and pushing
                clipped = np.clip(np.sign(reward), -1, 1)
                brain.buffer.push(state, action, clipped, next_state, float(done))
                loss, grad_norm = brain.train()

                #metrics
                total_reward += clipped
                steps += 1
                fail_safe += 1
                wandb.log({
                    "train/loss":          loss,
                    "train/grad_norm":     grad_norm,
                    "train/epsilon":       brain.eps,
                    "train/buffer_size":   len(brain.buffer),
                    "train/episode":         episode,
                    "train/learning_rate": brain.optimiser.param_groups[0]["lr"],
                }, step=steps)
                if done:
                    #time calcualtion 
                    ep_time = time.time() - ep
                    episode_time.append(ep_time)
                    eta = np.mean(episode_time[-100:]) * (args.episode - episode - 1)

                    #saving when necessary
                    if episode % args.mid_save == 0 and episode != 0: 
                        brain.save_checkpoint(episode, steps, arg_name)
                    if episode % args.full_save == 0 and episode != 0: 
                        brain.save()

                    pbar.set_postfix({
                        "ep": episode,
                        "loss": f"{loss:.4f}",
                        "reward": f"{total_reward:.1f}",
                        "eps": f"{brain.eps:.3f}",
                        "eta": format_time(eta),

                    })
                    pbar.update(1)

                    #updating the episode level metrics
                    wandb.log({
                        "episode/total_reward": total_reward,
                        "episode/RL_action_all": actions["all"],
                        "episode/RL_actions_up": actions["up"],
                        "episode/RL_actions_down": actions["down"],
                        "episode/RL_actions_neutral": actions["neutral"],
                        "episode/RB_action_all": robot.actions["all"],
                        "episode/RB_actions_up": robot.actions["up"],
                        "episode/RB_actions_down": robot.actions["down"],
                        "episode/RB_actions_neutral": robot.actions["neutral"],
                    }, step=steps)

                    #reseting

                    total_reward = 0
                    prev_action = 0
                    action = 0
                    robot.action = 0
                    actions = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
                    robot.actions = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
                    ep = time.time()
                    episode += 1
                    robot.reset()
                    fail_safe = 0
                    obs, _ = env.reset(seed=42)
                    if episode > 1 and episode%10==0 and args.camera:
                        switch = not switch
                        if switch:
                            print("using the camera")
                        else:
                            print("using in game info")
                    state = eye.reset() if switch else frame.reset(obs)
                state = next_state

    except KeyboardInterrupt:
        print("closing")
        env.close()
        act.join()
        controller.join()
        robot.inital()
        robot.rb.disconnect()
def eval(args):

    robot = Robot()

    #in case there is already saved positons use them
    if os.path.exists("positions.json"):
        with open("positions.json", "r") as f:
            temp = json.load(f)
        robot.positions = {keys:values for keys, values in temp.items()}
    
    #start the robot and put it at home
    robot.start()
    robot.inital()

    #pick the agent to test on 
    brain = Brain()
    brain.picking()
    brain.policy.eval()

    #use either camera or ingame obs
    print("using real camera") if args.camera else print("using in game info") 
    frame = Eyes() if args.camera else Frames()

    #initalize the env
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    state = frame.reset() if args.camera else frame.reset(obs)

    #start threading for reading the input, move the arm and playing the game
    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()

    #save the postions
    with open("positions.json", "w") as f:
        json.dump(robot.positions, f, indent=2)

    #render the screen on a specific display
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    count = 0
    total_reward = 0
    fail_safe = 0
    try:
        action = 0
        prev_action = 0
        while True:

            #get the action needed after completing the previous action
            if prev_action == robot.action or fail_safe >= 10:
                action = brain.rollout(state)
                fail_safe = 0
                if action == 2 or action ==4:
                    action = 2
                    robot.task = "up"
                elif action == 3 or action == 5:
                    action = 3
                    robot.task = "down"
                elif action == 0 or action == 1:
                    action = 0
                    robot.task = "neutral"
                else:
                    print('no clue')
                prev_action = action

            #move the state forward
            obs, reward, terminated, truncated, _ = env.step(robot.action)
            done = terminated or truncated
            state = frame.step() if args.camera else frame.step(obs)

            #render the state
            render = env.render()
            surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (1920, 1080))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(60)

            total_reward += reward
            
            fail_safe += 1
            if done:
                obs, _ = env.reset(seed=42)
                state = frame.reset() if args.camera else frame.reset(obs)
                print(f"{count+1}.Evaluating {brain.agent}, the score is : {total_reward}")
                count += 1 
                total_reward = 0
                action = 0
                prev_action = 0
                robot.reset()
                if count > args.iterations:
                    print("Done testing ", brain.agent)


    except KeyboardInterrupt:
        print("closing")
        env.close()
        act.join()
        controller.join()
        robot.inital()
        robot.rb.disconnect()
        
    
def main():
    #The argumaents provided in the code
    parser = argparse.ArgumentParser("Training DQN for the Robot Arm")
    parser.add_argument("-jn", "--job_name", help="Project name shown in wandb", type=str, default=str(random()))
    parser.add_argument("-ex", "--execution", help="The speed of the robot arm", type=float, default=config.EXECUTION)
    parser.add_argument("-thld", "--threshold", help="The threshold distance between the paddle and the middle of the screen", type=float, default=config.THRESHOLD)
    parser.add_argument("-ep", "--episode", help="The amount of episodes to train for in total", type=int, default=config.EPISODES)
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
    parser.add_argument("-tr", "--training", help="Toggle between training or eval", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-cam", "--camera", help="Toggle between using a camera or not", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-i", "--iterations", help="how many tests you want to do", default=config.ITERATION, type=int)
    
    args = parser.parse_args()

    if args.training:
        training(args)
    else:
        eval(args)


if __name__ == "__main__":
    main()