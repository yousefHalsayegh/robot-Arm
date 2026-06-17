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
from brain import Brain, Frames

gym.register_envs(ale_py)
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():

    env = gym.make("ALE/Pong-v5", frameskip=4)
    env.reset(seed=42)

    robot = Robot()

    if os.path.exists("positions.json"):
        with open("positions.json", "r") as f:
            temp = json.load(f)
        robot.positions = {keys:values for keys, values in temp.items()}
    
    robot.start()

    robot.inital()

    brain = Brain()
    path = ""
    frame = Frames()
    steps = 0
    start_time = time.time()
    episode_time = []

    if os.path.exists(path):
        _, _ = brain.load_checkpoint(path)
        print("loading... ", path )
    else:
        steps, start = 0, 0 

    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()
    with open("positions.json", "w") as f:
        json.dump(robot.positions, f, indent=2)
    try:
        for episode in range(start, config.EPISODES):
            obs, _ = env.reset()
            state = frame.reset(obs)
            total_reward = 0
            ep = time.time()
            
            goal_reward = 0
            tracking_reward = 0 
            prev_paddle_y = None
            action_lap = 0
            actions = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
            robot.actions = {"all": 0, "up" : 0, "down": 0, "neutral": 0}
            action = 0
            prev_action = 0
            while True:
                lap = time.time()

                if action_lap >= config.SLEEP:
                    action = brain.predict_next_action(state,steps, env)
                    if prev_action != action:
                        prev_action = action
                        if action == 2:
                            robot.task = "up"
                            actions['up'] += 1
                        elif action == 3:
                            robot.task = "down"
                            actions['down'] += 1
                        else:
                            robot.task = "neutral"
                            actions['neutral'] += 1
                        action_lap = 0
                        actions["all"] += 1

                obs, raw_reward, terminated, truncated, _ = env.step(robot.action)
                reward = 0
                done = terminated or truncated
                next_state = frame.step(obs)

                new_ball_y, new_paddle_y = brain.ball_position(next_state[-1])
                

                if raw_reward != 0:
                    goal = np.sign(raw_reward) * config.GOAL_REWARD
                    reward += goal
                    goal_reward += goal
                
        
                if new_ball_y is not None and new_paddle_y is not None and prev_paddle_y is not None:
                    new_distance = abs(new_ball_y - new_paddle_y)
                    prev_distance = abs(new_ball_y - prev_paddle_y)

                    if new_distance < prev_distance:
                        track = config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)
                        reward += track
                        tracking_reward += track
                    elif new_distance > config.THRESHOLD * config.CROP and new_distance >= prev_distance:
                        reward -= config.DISTANCE_REWARD * config.PENALTIY_MOVE
                        tracking_reward -= config.DISTANCE_REWARD * config.PENALTIY_MOVE
                    
                    center = config.PENALTIY_CENTER * ((abs(new_paddle_y - config.CENTER_Y))/config.CENTER_Y)
                    reward -= center
                    tracking_reward -= center

                prev_paddle_y = new_paddle_y

                brain.buffer.push(state, action, reward, next_state, float(done))
                state = next_state
                total_reward += reward
                steps += 1 

                loss = brain.train()
                if loss == None : 
                    loss = 0

                if done:
                    robot.reset()
                    break

                overall = max(0, config.SLOW- (time.time() - lap))
                time.sleep(overall)
                action_lap += overall

            ep_time = time.time() - ep
            episode_time.append(ep_time)
            eta = np.mean(episode_time[-100:]) * (config.EPISODES - episode - 1)
            print(f"Episode {episode} | Steps {steps} |  Loss {loss:.5f}")
            print(f"Total Reward {total_reward:.1f} | Tracking Reward {tracking_reward:.1f} |  Goal Reward {goal_reward:.1f} (Actual {(goal_reward/config.GOAL_REWARD):.1f})")
            print(f"Total Actions per Episode {actions['all']:.1f}/{robot.actions['all']:.1f} | Up Actions per Episode {actions['up']:.1f}/{robot.actions['up']:.1f} ")
            print(f"Neutral Actions per Episode {actions['neutral']:.1f}/{robot.actions['neutral']:.1f} | Down Actions per Episode {actions['down']:.1f}/{robot.actions['down']:.1f}")
            print(f"Episode time {format_time(ep_time)} | Total time {format_time(time.time() - start_time)} | ETA {format_time(eta)}")
        
            if episode % config.MID_SAVE == 0 and episode != 0: 
                brain.save_checkpoint(episode, steps, total_reward, tracking_reward, goal_reward, loss,"rob")
            if episode % config.FULL_SAVE == 0 and episode != 0: 
                brain.save()

    except KeyboardInterrupt:
        print("closing")
        env.close()
        act.join()
        controller.join()
        robot.rb.disconnect()
        
    



if __name__ == "__main__":
    main()