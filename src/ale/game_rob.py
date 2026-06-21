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
def picking():
    options = []
    for i in os.listdir():
        if os.path.exists(f"{i}/brain4900.pth"):
            options.append(f"{i}/brain4900.pth")

    print("Pick from the list which Agent you would like to evalute:")
    for i in range(len(options)):
        print(f"{i+1}.{options[i].split("/")[0]}")
    

    while True:
        try:
            choice = int(input()) - 1

            if choice > len(options) or choice < 0:
                print("Your option doesn't exist in the list, please pick something from the list")
                continue
            print("loading in ", options[choice])
            
            return options[choice]

        except ValueError:
            print("Please enter a number")
            continue
def main():

    robot = Robot()

    if os.path.exists("positions.json"):
        with open("positions.json", "r") as f:
            temp = json.load(f)
        robot.positions = {keys:values for keys, values in temp.items()}
    
    robot.start()

    robot.inital()
    brain = Brain()
    brain.load_checkpoint(picking())
    brain.policy.eval()
    choice = config.SWITCH
    if choice:
        print("using real camera")
        frame = Eyes()
    else:
        print("using in game info")
        frame = Frames()

    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    if choice:
        state = frame.reset()
    else:
        state = frame.reset(obs)

    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()
    with open("positions.json", "w") as f:
        json.dump(robot.positions, f, indent=2)

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=2)
    clock = pygame.time.Clock()
    try:
       while True:
            action = 0
            prev_action = 0
            while True:

                
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

                obs, _, terminated, truncated, _ = env.step(robot.action)
                done = terminated or truncated

                render = env.render()
                surf = pygame.surfarray.make_surface(render.transpose(1, 0, 2))
                scaled = pygame.transform.scale(surf, (1920, 1080))
                screen.blit(scaled, (0, 0))
                pygame.display.flip()
                clock.tick(60)


                if choice:
                    state = frame.step()
                else:
                    state = frame.step(obs)

                if done:
                    obs, _ = env.reset(seed=42)
                    choice != choice
                    print("switching to:")
                    if choice:
                        print("using real camera")
                        frame = Eyes()
                    else:
                        print("using in game info")
                        frame = Frames()
                    brain = Brain()
                    brain.load_checkpoint(picking())
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