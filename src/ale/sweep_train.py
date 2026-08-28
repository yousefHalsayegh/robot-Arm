"""
Entry point for a single wandb sweep run.
"""

import ast
import sys
import wandb
from game_rl import build_parser, training


def _extract_run_group_from_argv():
    """wandb agent passes --combo={...} as a literal dict string on the command line.
    Parse it here, before wandb.init(), since group= must be set at init time."""
    for arg in sys.argv[1:]:
        if arg.startswith("--combo="):
            combo_str = arg[len("--combo="):]
            try:
                combo = ast.literal_eval(combo_str)
                return combo.get("run_group")
            except (ValueError, SyntaxError):
                return None
    return None


def sweep_run():
    run_group = _extract_run_group_from_argv()
    wandb.init(group=run_group)   # group set here, at creation — this is the only valid time to set it

    args = build_parser().parse_args([])

    combo = dict(wandb.config.get("combo", {}))
    merged = {**combo, **{k: v for k, v in wandb.config.items() if k != "combo"}}

    unknown = [k for k in merged if not hasattr(args, k)]
    if unknown:
        raise ValueError(
            f"Sweep config contains parameter(s) not found in build_parser(): {unknown}. "
            f"Check for typos, or add them to build_parser()."
        )

    for key, value in merged.items():
        setattr(args, key, value)

    args.job_name = wandb.run.id
    args.wandb = True

    wandb.config.update(vars(args), allow_val_change=True)

    print(f"[sweep] resolved config for run {wandb.run.id} (group={run_group}):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    training(args)


if __name__ == "__main__":
    sweep_run()