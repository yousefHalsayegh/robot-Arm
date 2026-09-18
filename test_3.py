"""
check_camera_input.py

Standalone sense-check: launches Isaac Sim on its own, captures ONE real
RGB-D frame stack exactly the way the training loop does (Frames +
update_frame_stack), and saves the visualization — completely independent
of any training run. No train.py process needs to be running.

Run with:
    python check_camera_input.py --task <YOUR_TASK> --out camera_check.png --headless --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Standalone camera input sense-check")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, default="camera_check.png")
parser.add_argument("--settle_steps", type=int, default=60,
                     help="physics steps to run before capturing, so the scene "
                          "isn't caught mid-reset/falling into place")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from sim.tasks.joystick.mdp.observations import Frames, update_frame_stack
from visualize_camera_input import visualize_rgbd_stack


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    env.reset()
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())

    frame_stack = Frames(n=3)
    update_frame_stack(base_env, [frame_stack], reset_ids=[0])

    # step a bit so the frame stack fills with genuinely different moments
    # rather than n copies of the same just-reset frame
    for _ in range(args_cli.settle_steps):
        base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())
        simulation_app.update()
        update_frame_stack(base_env, [frame_stack])

    state = frame_stack._get_state()
    print(f"captured state shape: {state.shape}, dtype: {state.dtype}, "
          f"range: [{state.min():.3f}, {state.max():.3f}]")

    visualize_rgbd_stack(state, args_cli.out)

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)