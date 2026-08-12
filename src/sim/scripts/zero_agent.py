

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Joystick stiffness sweep")
parser.add_argument("--task",           type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--stiffness_values", type=float, nargs="+",
    default=[3, 3.5],
    help="candidate stiffness values to test, in the actuator's native units",
)
parser.add_argument(
    "--damping_values", type=float, nargs="+",
    default=[0.05],
    help="candidate damping values to test — every stiffness value is tested "
         "against every damping value (full grid, not paired 1:1)",
)
parser.add_argument("--grip_settle_steps", type=int, default=80,
                     help="steps to hold the grip and let drift settle before measuring")
parser.add_argument("--responsiveness_target", type=str, default="up",
                     help="which POSITIONS target to test responsiveness against")
parser.add_argument("--responsiveness_max_steps", type=int, default=20)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
import numpy as np
import torch
import gymnasium as gym

import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from sim.utils.robot_sim import POSITIONS, ARRIVAL_THRESHOLD, measure_joystick_response, RobotSim
from sim.tasks.joystick.mdp.rewards import DEADZONE_DEG, PIVOT_X_IDX, PIVOT_Y_IDX

def sim_step(base_env, simulation_app):
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    simulation_app.update()


def set_arm_joints(base_env, joint_positions: np.ndarray, device: str):
    """Teleport arm to exact joint positions instantly — used only for
    resetting between trials, not for the actual grip/response measurement."""
    robot = base_env.scene["robot"]
    target = torch.tensor(joint_positions, dtype=torch.float32, device=device).unsqueeze(0)
    zeros = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zeros)


def move_to_target(robot, target_pos: np.ndarray, device: str, sim_step_fn, n_steps: int):
    """Command a fixed target and let the PD controller settle toward it —
    same fixed-target mechanism used elsewhere in this project."""
    target_t = torch.tensor(target_pos, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(n_steps):
        robot.set_joint_position_target(target_t)
        robot.write_data_to_sim()
        sim_step_fn()


def reset_joystick(object_art, pivot_joint_ids, device: str):
    """Zero out the joystick's own joint state directly."""
    n_joints = object_art.data.joint_pos.shape[1]
    zero_pos = torch.zeros((1, n_joints), dtype=torch.float32, device=device)
    zero_vel = torch.zeros_like(zero_pos)
    object_art.write_joint_state_to_sim(zero_pos, zero_vel)


def measure_drift_deg(object_art, env_index: int = 0) -> float:
    tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())
    return float(np.max(np.abs(tilt_deg[[PIVOT_X_IDX, PIVOT_Y_IDX]])))


def main():
    
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = str(base_env.device)

    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    simulation_app.update()
    env.reset()
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())

    robot = base_env.scene["robot"]
    object_art = base_env.scene["object"]


    pivot_ids, pivot_names = object_art.find_joints(["PivotX", "PivotY"])
    print(f"found pivot joints: {pivot_names} at indices {pivot_ids}")

    dummy_robot = RobotSim(env_index=0)  # only needed to satisfy measure_joystick_response's interface

    results = []

    for stiffness in args_cli.stiffness_values:
      for damping in args_cli.damping_values:
        print(f"\n=== testing stiffness={stiffness} damping={damping} ===")

        # write the candidate stiffness/damping directly to the sim for this trial
        stiffness_t = torch.full((1, len(pivot_ids)), stiffness, dtype=torch.float32, device=device)
        damping_t   = torch.full((1, len(pivot_ids)), damping, dtype=torch.float32, device=device)
        object_art.write_joint_stiffness_to_sim(stiffness_t, joint_ids=pivot_ids)
        object_art.write_joint_damping_to_sim(damping_t, joint_ids=pivot_ids)

        # ── drift-under-grip test ──────────────────────────────────────────
        reset_joystick(object_art, pivot_ids, device)
        set_arm_joints(base_env, POSITIONS["home"], device)   # open, clear of joystick
        sim_step(base_env, simulation_app)

        move_to_target(robot, POSITIONS["neutral"], device,   # grips joystick at neutral
                       lambda: sim_step(base_env, simulation_app),
                       args_cli.grip_settle_steps)
        for _ in range(20):
            sim_step(base_env, simulation_app)
        drift_deg = measure_drift_deg(object_art, 0)
        print(f"  drift under grip: {drift_deg:.3f} deg")

        # release before the next test
        move_to_target(robot, POSITIONS["home"], device,
                        lambda: sim_step(base_env, simulation_app), 60)
        reset_joystick(object_art, pivot_ids, device)

        # ── responsiveness test ────────────────────────────────────────────
        move_to_target(robot, POSITIONS["neutral"], device,
                        lambda: sim_step(base_env, simulation_app), 60)
        for _ in range(20):
                    sim_step(base_env, simulation_app)
        steps_to_cross = measure_joystick_response(
            robot, object_art, dummy_robot,
            lambda: sim_step(base_env, simulation_app),
            target_name=args_cli.responsiveness_target,
            pivot_x_idx=PIVOT_X_IDX,
            deadzone_deg=DEADZONE_DEG,
            physics_dt=base_env.sim.get_physics_dt(),
            max_steps=args_cli.responsiveness_max_steps,
        )
        print(f"  steps to cross deadzone ({args_cli.responsiveness_target}): {steps_to_cross}")

        # release + reset for the next stiffness trial
        move_to_target(robot, POSITIONS["home"], device,
                        lambda: sim_step(base_env, simulation_app), 60)
        reset_joystick(object_art, pivot_ids, device)

        results.append({
            "stiffness": stiffness,
            "damping": damping,
            "drift_deg": drift_deg,
            "steps_to_cross_deadzone": steps_to_cross if steps_to_cross is not None else "never",
        })

    print("\n=== summary ===")
    print(f"{'stiffness':>10} {'damping':>10} {'drift_deg':>10} {'steps_to_cross':>15}")
    for r in results:
        print(f"{r['stiffness']:>10} {r['damping']:>10} {r['drift_deg']:>10.3f} {str(r['steps_to_cross_deadzone']):>15}")

    env.close()
    return 0


if __name__ == "__main__":
    main()

    simulation_app.close()