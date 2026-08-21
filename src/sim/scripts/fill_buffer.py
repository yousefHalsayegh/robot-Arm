import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Buffer pre-fill for low-level joystick SAC")
parser.add_argument("--task",              type=str, default=None)
parser.add_argument("--num_envs",          type=int, default=1)
parser.add_argument("--disable_fabric",    action="store_true", default=False)
parser.add_argument("--lerobot_repo_id",   type=str, default=None)
parser.add_argument("--output_path",       type=str, default="buffer_prefill_2.pkl")
parser.add_argument("--synthetic_per_cmd", type=int, default=100)
parser.add_argument("--interp_steps",      type=int, default=50)
parser.add_argument("--decision_steps",    type=int, default=10)
parser.add_argument("--action_scale_deg",  type=float, default=5.0)
parser.add_argument("--export_lerobot", default=True, action=argparse.BooleanOptionalAction)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    # imported here, AFTER AppLauncher is already running — safe
    from sim.utils.buffer import main as fill_buffer_main
    fill_buffer_main(args_cli, simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()