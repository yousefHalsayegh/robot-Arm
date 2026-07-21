
"""
the lerobot part
"""
import inputs
import time
import threading
import config

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.processor import make_default_processors

CONTROLLER = {
    # Keep in mind this is on the X/Y trgger not DP and the values are for Pong
    ('ABS_Y', 0): 2,   # up 
    ('ABS_Y',  255): 3,   # down
    ('ABS_Y',  128): 0,   # Neutral

}

class Robot():
    """
    Create a SO101 link without the VLA being present
    """
    def __init__(self):

        self.rb= make_robot_from_config(SOFollowerRobotConfig(
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
        ))
        #connects to the robot and initalize all the necessary info
        self.rb.connect()
        self.positions = {
            "home" : {},
            "neutral" : {},
            "up" : {},
            "down" : {}
        }
        self.reseting = False
        self.acting = False
        self.task = "neutral"
        self.action = 0
        self.prev = 0
        self.actions = {'all' : 0, 'neutral': 0, 'down': 0, 'up':0}
        
    def _wait_for_stick_confirm(self):
        
        while True:
            events = inputs.get_gamepad()
            for e in events:
                if e.code == 'ABS_Y':
                    return
    def controller(self):
        """
        Reads the actions from the controller
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
                    #done to measure action frequency vs called frequency from the RL
                    if self.prev != self.action:
                        self.prev = self.action
                        if act == 2:
                            self.actions['up'] += 1
                        elif act == 3:
                            self.actions['down'] += 1
                        else:
                            self.actions['neutral'] += 1
                        self.actions['all'] += 1
                            

            except Exception:
                break

    def start(self):
        """
        Used to save the location of each direction
        """
        tp = make_teleoperator_from_config(SOLeaderTeleopConfig(
            port="/dev/ttyLEADER",
            id="fighter_l"
        ))
        try:
            tp.connect()
            teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

                # drive the follower from the leader continuously in the
                # background while we wait for explicit capture confirmations.
                # duration is set very large to approximate "run until we're
                # done calibrating" -- teleop_loop wants a fixed duration, so
                # this is a workaround rather than a documented infinite mode.
            def teleop_worker():
                teleop_loop(
                    teleop=tp,
                    robot=self.rb,
                    fps=30,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    duration=10**7,
                )

            teleop_thread = threading.Thread(target=teleop_worker, daemon=True)
            teleop_thread.start()

            for i in ['up',  'neutral', 'down']:
                if not self.positions[i]:
                    print("Move the arm to the '", i, "' position with the leader,")
                    print("then flick the arcade stick to confirm capture.")
                    self._wait_for_stick_confirm()
                    obs = self.rb.get_observation()
                    self.positions[i] = {k:v for k,v in obs.items() if '.pos' in k}
                    print("Captured '", i, "':", self.positions[i])
                else:
                    print("already have position for ", i)

            
        finally:
            tp.disconnect()

    def act(self):
        """
        Sends the new coordinates depending on the task
        """
        while True:
            try:
                if self.reseting:
                    time.sleep(0.1)
                    continue
                self.rb.send_action(self.positions[self.task] )

            except Exception:
                break


    def inital(self):
        """
        Moves the robot arm toward home position
        """
        self.reseting = True
        time.sleep(config.EXECUTION + 0.1)
        home = self.positions['home']
        start = time.perf_counter()
        confirmed = 0 
        while(time.perf_counter() - start) < 10:
            self.rb.send_action(home)

            obs = self.rb.get_observation()
            error = max(abs(obs.get(k,0) - home[k]) for k in home)

            if error < 3:
                confirmed += 1
                if confirmed >= 5:
                    break
            
            time.sleep(config.EXECUTION)

        self.reseting = False

    def reset(self):
        """
        Moves the robot arm toward neutral positions
        """
        self.reseting = True
        time.sleep(config.EXECUTION + 0.1)
        neutral = self.positions['neutral']
        start = time.perf_counter()
        confirmed = 0 
        while(time.perf_counter() - start) < 10:
            self.rb.send_action(neutral)

            obs = self.rb.get_observation()
            error = max(abs(obs.get(k,0) - neutral[k]) for k in neutral)

            if error < 3:
                confirmed += 1
                if confirmed >= 5:
                    break
            
            time.sleep(config.EXECUTION)

        self.reseting = False
