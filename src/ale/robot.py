
"""
the lerobot part
"""
import config
import inputs
import random
import time

from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

CONTROLLER = {
    # D-pad vertical → paddle movement
    ('ABS_HAT0Y', -1): 2,   # up   → RIGHT (paddle up)
    ('ABS_HAT0Y',  1): 3,   # down → LEFT  (paddle down)
    ('ABS_HAT0Y',  0): 0,   # released → NOOP

    # D-pad horizontal (not used in Pong but mapped anyway)
    ('ABS_HAT0X', -1): 3,   # left
    ('ABS_HAT0X',  1): 2,   # right
    ('ABS_HAT0X',  0): 0,

    # Face buttons
    ('BTN_SOUTH', 1): 1,    # A → FIRE (serve)
    ('BTN_EAST',  1): 1,    # B → FIRE
    ('BTN_NORTH', 1): 1,    # Y → FIRE
    ('BTN_WEST',  1): 0,    # X → NOOP

    # Bumpers
    ('BTN_TR',  1): 2,      # RB → paddle up
    ('BTN_TL',  1): 3,      # LB → paddle down
    ('BTN_TR2', 1): 1,      # RT digital → FIRE
    ('BTN_TL2', 1): 1,      # LT digital → FIRE
}


class Robot():
    def __init__(self):

        cf = RobotClientConfig(
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
            actions_per_chunk=config.ACTION_PER_CHUNK ,
            task="up",
            policy_device="cuda",
            client_device="cuda",
            chunk_size_threshold= config.CHUNK_SIZE,
            aggregate_fn_name= config.AGGREGATE
        )

        self.client = RobotClient(cf)
        self.action = 1
        self.task = "up"
        self.show = False

    def controller(self):
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
        self.client.start_barrier.wait()
        while True:
            control_loop_start = time.perf_counter()
            try:
                if self.client.actions_available():
                    self.client.control_loop_action()
                
                if self.client._ready_to_send_observation():
                    self.client.control_loop_observation(self.task)


                if self.show:
                    print("current task ", self.task)
                    self.show = False
                time.sleep(max(0, (1/10) - (time.perf_counter() - control_loop_start)))
            except Exception:
                pass


    def update_task(self, new_task):
        if self.task != new_task:
            self.show = True
            self.task = new_task

            with self.client.action_queue_lock:
                self.client.action_queue.queue.clear()

            self.client.must_go.set()
