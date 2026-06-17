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


def main():

    env = gym.vector.SyncVectorEnv([env_init(42, i) for i in range(config.ENV)])

    brain = Brain()
    path = ""
    steps = 0
    start_time = time.time()
    episode_time = []

    if os.path.exists(path):
        steps, start = brain.load_checkpoint(path)
        print("loading... ", path )
    else:
        steps, start = 0, 0 

    try:
        episode = start
        obs, _ = env.reset()
        state = obs

        total_reward = np.zeros(config.ENV)
        ep = [time.time()] * config.ENV
        goal_reward = np.zeros(config.ENV)
        tracking_reward = np.zeros(config.ENV)
        prev_paddle_y = [None] * config.ENV
        action_lap = np.zeros(config.ENV)
        actions = [{"all": 0, "up" : 0, "down": 0, "neutral": 0} for i in range(config.ENV)]
        prev_action = np.zeros(config.ENV, dtype=np.int64)

        while episode < config.EPISODES:
            lap = time.time()

            ready = action_lap >= config.SLEEP

            action = brain.predict_next_action(state,steps, env)
            current_actions = np.where(ready, action, prev_action) #double check this bit
            for i in range(config.ENV):
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
            reward = np.zeros(config.ENV)
            done = terminated | truncated
            next_state = obs

            for i in range(config.ENV):
                new_ball_y, new_paddle_y = brain.ball_position(next_state[i][-1])
                

                if raw_reward[i] != 0:
                    goal = np.sign(raw_reward[i]) * config.GOAL_REWARD
                    reward[i] += goal
                    goal_reward[i] += goal
                
        
                if new_ball_y is not None and new_paddle_y is not None and prev_paddle_y[i] is not None:
                    new_distance = abs(new_ball_y - new_paddle_y)
                    prev_distance = abs(new_ball_y - prev_paddle_y[i])

                    if new_distance < prev_distance:
                        track = config.DISTANCE_REWARD * (prev_distance-new_distance / config.CROP)
                        reward[i] += track
                        tracking_reward[i] += track
                    elif new_distance > config.THRESHOLD * config.CROP and new_distance >= prev_distance:
                        reward[i] -= config.DISTANCE_REWARD * config.PENALTIY_MOVE
                        tracking_reward[i] -= config.DISTANCE_REWARD * config.PENALTIY_MOVE
                    
                    center = config.PENALTIY_CENTER * ((abs(new_paddle_y - config.CENTER_Y))/config.CENTER_Y)
                    reward[i] -= center
                    tracking_reward[i] -= center

                prev_paddle_y[i] = new_paddle_y

            for i in range(config.ENV):
                brain.buffer.push(state[i], current_actions[i], reward[i], next_state[i], float(done[i]))
            total_reward += reward
            steps += config.ENV
            
            for _ in range(config.UPDATES):
                loss = brain.train() or 0

            for i in np.where(done)[0]:
                ep_time = time.time() - ep[i]
                episode_time.append(ep_time)
                eta = np.mean(episode_time[-100:]) * (config.EPISODES - episode - 1)
                print(f"Environment {i} | Episode {episode} | Steps {steps} |  Loss {loss:.5f}")
                print(f"Total Reward {total_reward[i]:.1f} | Tracking Reward {tracking_reward[i]:.1f} |  Goal Reward {goal_reward[i]:.1f} (Actual {(goal_reward[i]/config.GOAL_REWARD):.1f})")
                print(f"Total Actions per Episode {actions[i]['all']:.1f}| Up {actions[i]['up']:.1f}| Neutral {actions[i]['neutral']:.1f}| Down {actions[i]['down']:.1f}")
                print(f"Episode time {format_time(ep_time)} | Total time {format_time(time.time() - start_time)} | ETA {format_time(eta)}")
            
                if episode % config.MID_SAVE == 0 and episode != 0: 
                    brain.save_checkpoint(episode, steps, total_reward, tracking_reward, goal_reward, loss,"rob")
                if episode % config.FULL_SAVE == 0 and episode != 0: 
                    brain.save()

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

            overall = max(0, config.SLOW- (time.time() - lap))
            time.sleep(overall)
            action_lap += overall

            action_lap[ready] = 0.0

       

    except KeyboardInterrupt:
        print("closing")
        env.close()
        
    



if __name__ == "__main__":
    main()