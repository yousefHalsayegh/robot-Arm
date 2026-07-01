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

import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from sim.utils.Pong import PongScreen

from ale.brain import Brain

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    base_env = env.unwrapped

    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    from pxr import UsdShade
    import isaacsim.core.utils.stage as stage_utils

    stage  = stage_utils.get_current_stage()
    shader = UsdShade.Shader.Get(stage, "/World/envs/env_0/screen/geometry/material/Shader")

    print("shader id:", shader.GetShaderId())
    for inp in shader.GetInputs():
        print(" input:", inp.GetFullName(), "|", inp.GetTypeName())
    # ── bind dynamic textures to screen prims ────────────────────────
    pong = PongScreen(
        num_envs=env_cfg.scene.num_envs,
        env_prim_prefix="/World/envs/env_",
        screen_relative_path="screen",
        seed=42,
    )
    pong.setup()
    # reset environment
    obs, _ = env.reset()
    steps = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
       
        env.step(0)
    
            

    # close the simulator
    env.close()
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()