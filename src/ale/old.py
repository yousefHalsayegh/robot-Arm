"""
the ALE part
"""

import sys
import time
import numpy as np
import gymnasium as gym
import ale_py
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.control_utils import predict_action, prepare_observation_for_inference
from lerobot.rollout.context import make_pre_post_processors
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLICY_PATH   = "/home/yousef/Documents/lerobot/outputs/train/smolvla_fighter_V1.63/checkpoints/030000/pretrained_model/"
ROBOT_PORT    = "/dev/ttyFOLLOWER"
ROBOT_ID      = "fighter_f"
CAMERA_SERIAL = "032522250421"
FPS           = 30

TASK_UP   = "up"
TASK_DOWN = "down"

_ALE_ACTION = {
    TASK_UP:   ale_py.Action.UP.value,
    TASK_DOWN: ale_py.Action.DOWN.value,
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

gym.register_envs(ale_py)

config = SOFollowerRobotConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    max_relative_target=10.0,
    use_degrees=True,
    cameras={
        "observation.images.front": RealSenseCameraConfig(
            serial_number_or_name=CAMERA_SERIAL,
            use_depth="true",
            width=1280,
            height=720,
            fps=FPS,
        )
    },
)
robot = SOFollower(config)
robot.connect()

policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
policy.eval()
device = torch.device(policy.config.device)

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=POLICY_PATH,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rollout_task(env, task, n_steps):
    print(f"\n[task={task}] running {n_steps} steps...")
    policy.reset()

    for _ in range(n_steps):
        t0 = time.perf_counter()

        # Step emulator
        _, _, terminated, truncated, _ = env.step(_ALE_ACTION[task])

        # Get raw observation from robot
        observation = robot.get_observation()

        # The policy expects:
        #   observation.state      -> np.ndarray shape (n_joints,)  all joint positions stacked
        #   observation.images.*   -> np.ndarray shape (H, W, 3)    camera frames unchanged
        # robot.get_observation() returns individual float scalars per joint, so stack them.
        joint_keys = sorted(
            k for k in observation
            if not k.startswith("observation.images.")
        )
        state = np.array([
            observation[k] if not isinstance(observation[k], np.ndarray) else observation[k].item()
            for k in joint_keys
        ], dtype=np.float32)
        observation = {
            "observation.state": state,
            **{k: v for k, v in observation.items() if k.startswith("observation.images.")},
        }

        # predict_action handles: prepare obs → normalize → policy → denormalize
        action_values = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=task,
            robot_type=robot.robot_type,
        )

        # action_values is a 1D tensor of shape (n_joints,)
        action_values = action_values.flatten()
        action = {
            key: action_values[i].item()
            for i, key in enumerate(robot.action_features)
        }

        robot.send_action(action)

        joints = "  ".join(f"{k.split('.')[0][:4]}={v:+.1f}" for k, v in action.items())
        print(f"\r[{task}] {joints}   ", end="", flush=True)

        if terminated or truncated:
            env.reset()
            policy.reset()

        time.sleep(max(0, 1.0 / FPS - (time.perf_counter() - t0)))


def main():
    env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=8)
    env.reset(seed=42)

    task = TASK_UP
    STEPS_PER_INPUT = 100

    print("=" * 50)
    print(f"Current task: {task}")
    print("Enter to continue | up/down + Enter to switch | q to quit")
    print("=" * 50)

    try:
        while True:
            rollout_task(env, task, STEPS_PER_INPUT)

            print(f"\n[input] current={task}: ", end="", flush=True)
            val = sys.stdin.readline().strip().lower()

            if val == "q":
                break
            elif val in (TASK_UP, TASK_DOWN):
                task = val
                print(f"[input] task → {task}")

    finally:
        robot.disconnect()
        env.close()


if __name__ == "__main__":
    main()