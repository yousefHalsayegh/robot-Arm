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

gym.register_envs(ale_py)
def main():

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
    choice = config.SWITCH
    print("using real camera") if choice else print("using in game info") 
    frame = Eyes() if choice else Frames()

    #initalize the env
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    state = frame.reset() if choice else frame.reset(obs)

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
    try:
       while True:
            action = 0
            prev_action = 0
            while True:

                #get the action needed after completing the previous action
                if prev_action == robot.action:
                    action = brain.rollout(state)
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
                obs, _, terminated, truncated, _ = env.step(robot.action)
                done = terminated or truncated

                #render the state
                render = env.render()
                surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
                scaled = pygame.transform.scale(surf, (1920, 1080))
                screen.blit(scaled, (0, 0))
                pygame.display.flip()
                clock.tick(60)


                state = frame.reset() if choice else frame.reset(obs)

                if done:
                    obs, _ = env.reset(seed=42)
                    choice = not choice
                    #restart, change the current way of obs and pick a new model to eval
                    print("switching to camera setting")
                    frame = Eyes() if choice else Frames()
                    brain = Brain()
                    brain.picking()
                    brain.policy.eval()


    except KeyboardInterrupt:
        print("closing")
        env.close()
        act.join()
        controller.join()
        robot.inital()
        robot.rb.disconnect()
        
    



if __name__ == "__main__":
    main()