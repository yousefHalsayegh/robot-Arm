#TODO needs change and update to fit the other ones 

import ale_py
import gymnasium as gym
import threading
import time
import json
import config
import random
from robot import Robot

def finetune_list():
    f = {}
    chunk = [25, 50, 75, 100]
    threshold = [0.25, 0.50, 0.75, 1.0]
    aggregation = ["weighted_average", "latest_only", "average", "conservative"]
    for c in chunk:
        for t in threshold:
            f[(c,t)] = {
                "down to up" : [],
                "up to down" : [],
                "up": [],
                "down" :[]
            }

    return f 

gym.register_envs(ale_py)

def remap_keys(mapping):
    return [{'key':k, 'value': v} for k, v in mapping.iteritems()]

def main():

    with open("finetune.json", "r") as f :
        temp = json.load(f)
    finetune = {(float(keys[1:4]), float(keys[-4:-1])):values for keys, values in temp.items()}
    keys = list(finetune.keys())
    print(keys)
    random.shuffle(keys)

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
    did = set()
    try:
        for k in keys:
            if len(finetune[k]["up"]) > 0 or len(finetune[k]["down"]) > 0 or len(finetune[k]["up to down"]) > 0 or len(finetune[k]["down to up"]) > 0:
                print(k, " already has information moving to the next one")
                continue
            print("doing currently ", k)
            robot.client.action_chunk_size = k[0]
            robot.client._chunk_size_threshold = k[1]
            current = [0, 0, 0 , 0]
            count = 0 
            for _ in range(config.EPISODES):
                _, _ = env.reset()
                start = time.time()
                print(f"now working on size:{robot.client.action_chunk_size} and threshold:{robot.client._chunk_size_threshold} with action {robot.task}")

                while True:
                    t_start = time.perf_counter()

                    _, _, terminated, truncated, _ = env.step(robot.action)
                    done = terminated or truncated

                    if (robot.task == "up" or robot.task == "down to up") and robot.action == 2:
                        finetune[k][robot.task].append(abs(start - time.time()))
                        print(f"for {k} and task '{robot.task}' it took {finetune[k][robot.task][-1]} and the current amount is {len(finetune[k][robot.task])}")
                        start = time.time()
                        robot.update_task("up to down")
                        print("doing the following task ", robot.task)

                    elif (robot.task == "down" or robot.task == "up to down") and robot.action == 3:
                        finetune[k][robot.task].append(abs(start - time.time()))
                        print(f"for {k} and task '{robot.task}' it took {finetune[k][robot.task][-1]} and the current amount is {len(finetune[k][robot.task])}")
                        start = time.time()
                        robot.update_task("down to up")
                        print("doing the following task ", robot.task)

                    
                    elapsed = time.perf_counter() - t_start
                    time.sleep(max(0, (1/10) - elapsed))

                    
                    if config.TIMEOUT < abs(start - time.time()):
                        start = time.time()
                        robot.reset()
                        break 

                    if done:
                        print("done with the episode")
                        with open("finetune.json", "w") as f :
                            json.dump({str(key): value for key, value in finetune.items()}, f, indent=2)
                        robot.reset()
                        break

                   

                if current[0] == len(finetune[k]["down to up"]) and current[1] == len(finetune[k]["up to down"]) and current[2] == len(finetune[k]["up"]) and current[3] == len(finetune[k]["down"]):
                    count += 1 
                    print("counting since no change")
                else:
                    current[0] = len(finetune[k]["down to up"])
                    current[1] = len(finetune[k]["up to down"])
                    current[2] = len(finetune[k]["up"])
                    current[3] = len(finetune[k]["down"])

                if (abs(len(finetune[k]["up to down"]) -len(finetune[k]["down to up"])) >= 5) or (abs(len(finetune[k]["up"]) -len(finetune[k]["down"])) >= 5):
                    count += 1 
                    print("counting since the change is big")
                if len(finetune[k]["up to down"]) >= config.LENGTH and len(finetune[k]["down to up"]) >= config.LENGTH:
                    print("reached the limit")
                    with open("finetune.json", "w") as f :
                        json.dump({str(key): value for key, value in finetune.items()}, f, indent=2)
                    robot.reset()
                    break 
                if count > config.STEPS:
                    print("nothing changed for too long")
                    with open("finetune.json", "w") as f :
                        json.dump({str(key): value for key, value in finetune.items()}, f, indent=2)
                    robot.reset()
                    break
            if k not in did:
                did.add(k)
                print("currently done are: ", did)

    except KeyboardInterrupt:
        print("closing")
        env.close()
        action_receiver_thread.join()
        playing.join()
        controller.join()
        robot.client.stop()
        
    



if __name__ == "__main__":
    main()