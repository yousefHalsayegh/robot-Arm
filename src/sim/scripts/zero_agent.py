# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import carb.input
import omni.appwindow
import torch
from ale.brain import Brain



def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    
    robot = env.unwrapped.scene["robot"]
    joy = env.unwrapped.scene["object"]
    STEP = 0.05  # radians per keypress
    targets = robot.data.default_joint_pos.clone()
    selected = [0]
    saved_poses = {}
    sim_env = env.unwrapped

    for i in range(10):
        sim_env.sim.step(render=True)
        joy.update(sim_env.physics_dt)
        print(i, "target:", joy.data.joint_pos_target[0].cpu().numpy(),
                "actual:", joy.data.joint_pos[0].cpu().numpy())
    print(joy.data.joint_stiffness)
    print(joy.data.joint_damping)
    print(joy.joint_names)
    def on_key(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name
            if name[-1].isdigit() and int(name[-1]) < robot.num_joints:
                selected[0] = int(name[-1])
                print(f"Selected joint {selected[0]}: {robot.joint_names[selected[0]]}")
            elif name == "UP":
                targets[:, selected[0]] += STEP
            elif name == "DOWN":
                targets[:, selected[0]] -= STEP
            elif name == "P":
                current_deg = np.rad2deg(robot.data.joint_pos[0].cpu().numpy())
                print("Current joint_pos (deg):")
                for jname, val in zip(robot.joint_names, current_deg):
                    print(f"  {jname}: {val}")

            elif name == "S":
                pose_name = input("Save current pose as: ")
                current_deg = np.rad2deg(robot.data.joint_pos[0].cpu().numpy())
                saved_poses[pose_name] = current_deg.tolist()
                print(f"Saved '{pose_name}' , {current_deg}")

            elif name == "D":
                # dump everything saved so far, formatted like POSITIONS
                print("\nPOSITIONS = {")
                for pname, vals in saved_poses.items():
                    print(f'    "{pname}": np.array([')
                    for jname, v in zip(robot.joint_names, vals):
                        print(f"        {v!r},  # {jname}")
                    print("    ], dtype=np.float32),\n")
                print("}\n")
        return True
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, on_key)

    # then in the loop
    while simulation_app.is_running():
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        robot.update(sim_env.physics_dt)  
        simulation_app.update()
            

    # close the simulator
    env.close()
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()