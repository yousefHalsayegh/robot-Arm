"""
the ALE part playing using the inputs
"""

import ale_py
import gymnasium as gym
import threading
import time

from robot import Robot



gym.register_envs(ale_py)

def main():
    env = gym.make("ALE/Pong-v5", frameskip=4, render_mode="human")
    env.reset(seed=42)

    robot = Robot()
 
    robot.client.start()

    action_receiver_thread = threading.Thread(target=robot.client.receive_actions, daemon=True)
    action_receiver_thread.start()

    playing = threading.Thread(target=robot.controller, daemon=True)
    playing.start()

    controller = threading.Thread(target=robot.send, daemon=True)
    controller.start()

    start = time.time()

    try:
        while True:
            t_start = time.perf_counter()
            
            obs, raw_reward, terminated, truncated, _ = env.step(robot.action)
            done = terminated or truncated

            if (robot.task == "up" or robot.task == "down to up") and robot.action == 2:
                robot.update_task("up to down")
                print(time.time() - start, "now executing", robot.task, " show ", robot.show)
                start = time.time()

            elif (robot.task == "down" or robot.task == "up to down") and robot.action == 3:
                robot.update_task("down to up")
                print(time.time() - start, "now executing", robot.task, " show ", robot.show)
                start = time.time()
            
            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, (1/10) - elapsed))

            
    finally:
        env.close()
        robot.client.stop()
        action_receiver_thread.join()
        playing.join()
        controller.join()



if __name__ == "__main__":
    main()