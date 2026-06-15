"""
the ALE part training the robot
"""

import ale_py
import gymnasium as gym
import threading
import time
import random
import config
from robot_noVla import Robot
from brain import Brain, Frames

gym.register_envs(ale_py)


def main():

    env = gym.make("ALE/Pong-v5", frameskip=4, render_mode="human")
    env.reset(seed=42)

    robot = Robot()
    robot.start()

    
    print("reseting the environemt")
    robot.reset()

    brain = Brain()
    frame = Frames()

    controller = threading.Thread(target=robot.controller, daemon=True)
    controller.start()

    act = threading.Thread(target=robot.act, daemon=True)
    act.start()
    try:
        for _ in range(config.EPISODES):
            obs, _ = env.reset()
            state = frame.reset(obs)

            while True:
                act = brain.rollout(state)
                if act == 2:
                    print("I am yelling up")
                    robot.task = "up"
                elif act == 3:
                    print("I am yelling down")
                    robot.task = "down"
                else:
                    print("I am yelling neutral ", act)
                    robot.task = "neutral"

                obs, _, terminated, truncated, _ = env.step(robot.action)
                done = terminated or truncated
                state = frame.step(obs)

               
                if done:
                    print("done with the episode")
                    robot.reset()
                    break

    except KeyboardInterrupt:
        print("closing")
        env.close()
        act.join()
        controller.join()
        robot.rb.disconnect()
        
    



if __name__ == "__main__":
    main()