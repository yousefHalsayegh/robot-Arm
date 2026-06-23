
"""
the lerobot part
"""
import config
import inputs
import time
import random 

from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

CONTROLLER = {
    # Keep in mind this is on the X/Y trgger not DP and the values are for Pong
    ('ABS_Y', 0): 2,   # up 
    ('ABS_Y',  255): 3,   # down
    ('ABS_Y',  128): 0,   # Neutral

}


class Robot():
    """
    Create the connection with SO101 arm, while using a VLA 
    """
    def __init__(self, init_action_per_chunk=config.ACTION_PER_CHUNK, init_chunk_size_threshold = config.CHUNK_SIZE, init_aggregate_fn_name = config.AGGREGATE):
        #initializing the connection with the SO101
        rcf = RobotClientConfig(
            policy_type="smolvla",
            pretrained_name_or_path=config.POLICY,
            robot=SOFollowerRobotConfig(
                port="/dev/ttyFOLLOWER",
                id="fighter_f",
                cameras={
                    "camera1": RealSenseCameraConfig(
                        serial_number_or_name="032522250421",
                        use_depth="true",
                        width=1280,
                        height=720,
                        fps=30
                    )
                }
            ),
            actions_per_chunk=init_action_per_chunk ,
            task="up",
            policy_device="cuda",
            client_device="cuda",
            chunk_size_threshold= init_chunk_size_threshold,
            aggregate_fn_name= init_aggregate_fn_name
        )

        self.client = RobotClient(rcf)

        #Starting information, to ensure the robot is ready 
        self.action = 1
        self.task = random.choice(["up", "down"])
        home_obs = self.client.robot.get_observation()
        self.home_position = {k: v for k, v in home_obs.items() if '.pos' in k}
        self.reseting = False

    def controller(self):
        """
        Used to read the input from the Arcade stick
        """
        while True:
            try:
                events = inputs.get_gamepad()

                for e in events:
                    if e.ev_type in ("Sync", "Misc"):
                        continue
                    if e.code in ('ABS_GAS', 'ABS_BRAKE'):
                        continue

                    key = (e.code, e.state)
                    act = CONTROLLER.get(key)

                    self.action = act

            except Exception:
                pass


    def send(self):
        """
        Send observaton to the server to collect actions and execute the actions
        """

        self.client.start_barrier.wait()
        while True:
            control_loop_start = time.perf_counter()
            try:
                if self.reseting:
                    time.sleep(0.05)
                    continue

                if self.client.actions_available():
                    self.client.control_loop_action()
                
                if self.client._ready_to_send_observation():
                    self.client.control_loop_observation(self.task)

                time.sleep(max(0, (1/10) - (time.perf_counter() - control_loop_start)))
            except Exception:
                pass


    def update_task(self, new_task):
        """
        Changes the task as long as there is a new task
        """
        if self.task != new_task:
            self.task = new_task

    def flush(self):
        """
        FLush out the server side so that no conflicting actins are present
        """
        with self.client.action_queue_lock:
            self.client.action_queue.queue.clear()

        self.client.must_go.set()

    def reset(self):
        """
        Return the robot arm to home positions
        """
        self.reseting = True

        time.sleep(0.1)
        self.flush()
        TOLERANCE = 5.0
        TIMEOUT   = 10.0
        GRIPPER = 35.0
        
        #reads the current location of the arm
        obs     = self.client.robot.get_observation()
        current = {k: obs[k] for k in self.home_position if k in obs}
        current["gripper.pos"] = GRIPPER
        #sends to only move the gripper so that nothing breaks
        self.client.robot.send_action(current)
        time.sleep(1.0)

        #keeping the gripper location in mind send to move to home positon
        obs     = self.client.robot.get_observation()
        home_cmd = {**self.home_position, "gripper.pos": GRIPPER}

        t_start = time.perf_counter()
        while (time.perf_counter() - t_start) < TIMEOUT:

            self.client.robot.send_action(home_cmd)
            obs          = self.client.robot.get_observation()
            error   = max(abs(obs.get(k, 0) - self.home_position[k]) for k in self.home_position)
            if error < TOLERANCE:
                self.flush()
                self.reseting = False
                return 
            time.sleep(1)


        self.flush()
        self.reseting = False
        print("Reset timeout — continuing anyway")