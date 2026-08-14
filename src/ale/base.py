"""
the ALE part training the robot
"""

import ale_py
import gymnasium as gym
import os
import json
from robot_noVla import Robot
import numpy as np
import threading
import pygame
import time

gym.register_envs(ale_py)


NOOP, FIRE, UP, DOWN = 0, 1, 2, 3

def ball_position(obs):
        """
        Used for Atari Pong, using the observation of the env calculates the ball and paddle position. 
        """

        #divides the screen into where the court (mid side) and player (right side) of the screens
        court =obs[15:77, 12:71]
        player = obs[15:77, 72:76]

        #Locates the location of the ball using thresholds for the intensity then extracting the Y axis
        ball_pixels = np.argwhere((court > 0.4) & (court < 0.9))
        ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None



        #Locates the location of the player paddle using thresholds for the intensity then extracting the Y axis
        paddle_pixels = np.argwhere((player > 0.5) & (player < 0.7))
        paddle_y = float(np.mean(paddle_pixels[:, 0])) if len(paddle_pixels) > 0 else None
        return ball_y, paddle_y

def choose_action(ball_y, paddle_y, deadzone=2.5):
    """
    Simple proportional controller: move the paddle toward the ball's y position.
    """
    if ball_y is None or paddle_y is None:
        return FIRE  # nothing detected (e.g. ball hasn't been served yet) -> serve
 
    diff = ball_y - paddle_y
    if abs(diff) < deadzone:
        return NOOP
    elif diff < 0:
        return UP
    else:
        return DOWN


def main():
    robot = Robot()
    #in case there is already saved positons use them
    if os.path.exists("positions.json"):
        with open("positions.json", "r") as f:
            temp = json.load(f)
        robot.positions = {keys:values for keys, values in temp.items()}
    
    #start the robot and put it at home
    robot.inital()

    #initalising the env
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array", repeat_action_probability=0)
    env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True,
        )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    obs, _ = env.reset(seed=42)
    
    #rendering the info in the proper screen for the camera
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    total_reward = 0
    action = 0

    #start threading for reading the input, move the arm and playing the game
    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()

    #save the postions
    with open("positions.json", "w") as f:
        json.dump(robot.positions, f, indent=2)
    start = time.perf_counter()
    actions = {
        0: "neutral",
        1: "neutral",
        2: "up",
        3: "down",
        4 : "up",
        5 : "down"
    }
    while True:
        #main loop
        try:


            #moving the state forward and rendering it 
            obs, reward, terminated, truncated, _ = env.step(robot.action)
            done = terminated or truncated
            # if total_reward == 0:
            #     debug_calibrate(np.asarray(obs)[-1])

            new_ball_y, new_paddle_y = ball_position(obs[-1])
            action = choose_action(new_ball_y, new_paddle_y)
            robot.task = actions[action]
    
            render = env.render()
            surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (1920, 1080))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(10)
            total_reward += reward
            
            if done:
                obs, _ = env.reset(seed=42)
                
                
        except KeyboardInterrupt:
            print("\nclosing...")
            env.close()
            break

if __name__ == "__main__":
    main()