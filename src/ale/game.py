"""
the ALE part training RL
"""

import time
import numpy as np
import gymnasium as gym
import ale_py
import cv2
from collections import deque
import os
import torch

from brain import Brain
import config

class Frames():
    def __init__(self, n=4):
        self.frames = deque(maxlen=n)
        self.n = n

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84,84))
        img = img/255.0

        return img.astype(np.float32)
    
    def reset(self, img):
        proc = self.preprocess(img)
        for _ in range(self.n):
            self.frames.append(proc)
        return self._get_state()
    
    def step(self, img):
        self.frames.append(self.preprocess(img))
        return self._get_state()
    
    def _get_state(self):
        return np.stack(self.frames, axis=0)

gym.register_envs(ale_py)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def ball_position(obs):

    court =obs[14:76, 16:79]
    player = obs[14:76, 70:79]
    ball_pixels = np.argwhere((court > 0.7) & (court < 0.9))

    ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None

    paddle_pixels = np.argwhere((player > 0.4) & (player < 0.9))

    paddle_y = float(np.mean(paddle_pixels[:, 0])) if len(paddle_pixels) > 0 else None

    return ball_y, paddle_y

def main():
    env = gym.make("ALE/Pong-v5", frameskip=4)
    env.reset(seed=42)
    frame = Frames()
    brain = Brain()
    steps = 0
    start_time = time.time()
    episode_time = []
    path = "checkpoint/brain1300.pth"
    if os.path.exists(path):
        steps, start = brain.load_checkpoint(path)
        print("loading... ", path )
    else:
        steps, start = 0, 0 


    try:
        for episode in range(start, config.EPISODES):

            obs,_ = env.reset()
            state = frame.reset(obs)
            total_reward = 0
            ep = time.time()
            
            goal_reward = 0
            tracking_reward = 0 
            prev_paddle_y = None


            while True:

                action = brain.predict_next_action(state, steps, env)
                obs, raw_reward, terminated, truncated, _ = env.step(action)
                reward = 0
                
                done = terminated or truncated

                next_state = frame.step(obs)

                new_ball_y, new_paddle_y = ball_position(next_state[-1])
                

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
                #replay
                state = next_state
                total_reward += reward
                steps += 1 

                loss = brain.train()
                if loss == None : 
                    loss = 0

                if done:
                    break
            ep_time = time.time() - ep
            episode_time.append(ep_time)
            eta = np.mean(episode_time[-100:]) * (config.EPISODES - episode - 1)
            print(f"Episode {episode} | Steps {steps} | Total Reward {total_reward:.1f} | Tracking Reward {tracking_reward:.1f} |  Goal Reward {goal_reward:.1f} (Actual {(goal_reward/config.GOAL_REWARD):.1f})| Loss {loss:.5f}")
            print(f"Episode time {format_time(ep_time)} | Total time {format_time(time.time() - start_time)} | ETA {format_time(eta)}")
        
            if episode % config.MID_SAVE == 0 and episode != 0: 
                brain.save_checkpoint(episode, steps)
            if episode % config.FULL_SAVE == 0 and episode != 0: 
                brain.save()

    finally:
        brain.save()
        env.close()



if __name__ == "__main__":
    main()