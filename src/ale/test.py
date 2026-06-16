import time
import os
import json
from robot_noVla import Robot
import threading

def measure_execution_time(robot, task):
    t_start = time.perf_counter()
    robot.rb.send_action(robot.positions[task])
    start = time.perf_counter()
    while (time.perf_counter() - start) < 10:
        obs   = robot.rb.get_observation()
        error = max(
            abs(obs.get(k, 0) - robot.positions[task][k])
            for k in robot.positions[task]
        )
        if error < 5 and (task == "neutral" or task =="home") :
            break
        if robot.action != 0 :
            robot.action = 0 
            print("did the action")
            break
        
    
    elapsed = time.perf_counter() - t_start
    print(f"Execution time: {elapsed:.3f}s")


robot = Robot()

if os.path.exists("positions.json"):
    with open("positions.json", "r") as f:
        temp = json.load(f)
    robot.positions = {keys:values for keys, values in temp.items()}

robot.start()
controller = threading.Thread(target=robot.controller, daemon=True)
controller.start()

for i in ['home', 'neutral', 'up', 'neutral', 'down', 'up', 'down', 'neutral', 'home']:
    print("doing ", i)
    measure_execution_time(robot, i)