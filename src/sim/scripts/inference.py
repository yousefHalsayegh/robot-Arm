"""Evaluate one or more trained checkpoints.

Loads each checkpoint's own saved architecture metadata (use_camera,
num_commands, active_commands) so every model is reconstructed to exactly
match what it was trained with, runs deterministic rollouts for a target
number of episodes per model, and reports per-command success rate plus the
min / max / mean of that rate across the model's active commands, alongside
the overall aggregate success rate.

The environment is built once (with the camera scene present) and reused
across every checkpoint, regardless of that checkpoint's own use_camera
flag — a camera-off model simply never reads the observation, so there's no
correctness cost, and it avoids relaunching Isaac Sim per model.

Run with:
    python eval.py --task <YOUR_TASK> --checkpoints ckpt_a.pth ckpt_b.pth \
        --episodes_per_model 200 --num_envs 16 --headless --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Model evaluation")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                     help="one or more .pth checkpoint paths to evaluate directly. "
                          "Optional if --summary is given with no --checkpoints_file "
                          "either — in that case checkpoints are derived automatically "
                          "from the summary's own job names (each job's "
                          "runs/LowLevel-<job_name>/Full/manipulation_brain_*.pth).")
parser.add_argument("--checkpoints_file", type=str, default=None,
                     help="path to a text file listing checkpoints to evaluate, one per "
                          "line. Blank lines and lines starting with '#' are ignored. Each "
                          "line may be a literal .pth path, a glob pattern (e.g. "
                          "'runs/*/Full/*.pth'), or a directory — a directory expands to "
                          "every .pth file directly inside it. Combined with --checkpoints "
                          "if both are given.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes_per_model", type=int, default=200,
                     help="total episodes to run per model, split across num_envs")
parser.add_argument("--eval_episode_length_s", type=float, default=10.0,
                     help="fixed episode length used for every model — no curriculum during eval")
parser.add_argument("--eval_decision_steps", type=int, default=50,
                     help="fixed decision-window length used for every model during eval")
parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction,
                     help="use the actor's deterministic (mean) action rather than sampling")
parser.add_argument("--output_csv", type=str, default=None,
                     help="optional path to write a per-model, per-command CSV of results")
parser.add_argument("--summary", type=str, default=None,
                     help="optional path to an ablation_summary.md — used as a fallback "
                          "source for active_commands when a checkpoint has no saved "
                          "metadata and no explicit override was given in the checkpoints file, "
                          "and as the source of wandb run URLs for --download_history")
parser.add_argument("--download_history", default=True, action=argparse.BooleanOptionalAction,
                     help="before evaluating checkpoints, download each job's full wandb "
                          "training history (via scan_history — unsampled, full fidelity) "
                          "into --history_out_dir/<job_name>.csv. Requires --summary to "
                          "locate each job's wandb run.")
parser.add_argument("--history_out_dir", type=str, default="eval",
                     help="output directory for downloaded per-job wandb history CSVs")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import glob
import os
import re
import numpy as np
import torch
import wandb
import gymnasium as gym
from tqdm import tqdm
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from sim.utils.robo_brain import Brain
from sim.utils.robot_sim import ARRIVAL_THRESHOLD, POSITIONS
from sim.tasks.joystick.mdp.observations import Frames, update_frame_stack
from sim.tasks.joystick.play_env_cfg import ALL_COMMANDS, CMD_NEUTRAL, CMD_HOME


def resolve_checkpoints_from_summary(summary_path: str):
    """Derives the checkpoint list directly from ablation_summary.md's own
    job names, rather than requiring a separately-authored checkpoints file.
    For each job name found in the summary, checks
    runs/LowLevel-<job_name>/Full/manipulation_brain_*.pth — the same path
    and glob pattern the ablation sweep itself uses to decide a job is
    already complete. Jobs with no Full checkpoint (dependency-skipped, or
    not yet run to completion) are skipped with a printed note, not an
    error. Returns the same (path, override) pair format as
    resolve_checkpoint_list, with override always None here since this path
    has no per-line override syntax to parse."""
    if not summary_path or not os.path.exists(summary_path):
        raise FileNotFoundError(f"--summary not found: {summary_path}")

    job_re = re.compile(r"^\-\s+\*\*(.+?)\*\*")
    job_names = []
    with open(summary_path) as f:
        for line in f:
            m = job_re.match(line.strip())
            if m:
                job_names.append(m.group(1))

    resolved = []
    for job_name in job_names:
        full_dir = f"runs/LowLevel-{job_name}/Full"
        matches = sorted(glob.glob(os.path.join(full_dir, "manipulation_brain_*.pth")))
        if not matches:
            print(f"[skip] {job_name}: no Full checkpoint at {full_dir} — "
                  f"job was likely dependency-skipped or never completed training")
            continue
        # if more than one somehow exists, take the highest episode number
        def _episode_num(p):
            m = re.search(r"manipulation_brain_(\d+)\.pth", p)
            return int(m.group(1)) if m else -1
        latest = max(matches, key=_episode_num)
        resolved.append((latest, None))

    if not resolved:
        raise ValueError(f"no Full checkpoints found for any job listed in {summary_path}")
    return resolved


def resolve_checkpoint_list(direct_checkpoints, checkpoints_file):
    """Builds the final, deduplicated, order-preserving list of
    (checkpoint_path, override_commands_str_or_None) pairs from --checkpoints
    and/or --checkpoints_file. Each entry may optionally end in
    '|<comma-separated command names or "all">' to explicitly declare the
    active commands for that path (or every path a glob/directory entry
    expands to) — e.g. 'runs/foo/*.pth|up,down'. Each entry (before the
    override) may be a literal path, a glob pattern, or a directory
    (expanded to its top-level *.pth files)."""
    raw_entries = [(c, None) for c in direct_checkpoints] if direct_checkpoints else []

    if checkpoints_file:
        if not os.path.exists(checkpoints_file):
            raise FileNotFoundError(f"--checkpoints_file not found: {checkpoints_file}")
        with open(checkpoints_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    entry, override = line.split("|", 1)
                    raw_entries.append((entry.strip(), override.strip()))
                else:
                    raw_entries.append((line, None))

    resolved = []   # list of (path, override_or_None)
    seen = set()
    for entry, override in raw_entries:
        if os.path.isdir(entry):
            matches = sorted(glob.glob(os.path.join(entry, "*.pth")))
            if not matches:
                print(f"[warn] directory '{entry}' contains no .pth files — skipping")
        elif any(ch in entry for ch in "*?["):
            matches = sorted(glob.glob(entry))
            if not matches:
                print(f"[warn] glob pattern '{entry}' matched nothing — skipping")
        else:
            matches = [entry]

        if override and len(matches) > 1:
            print(f"[note] override '{override}' from entry '{entry}' will be applied "
                  f"to all {len(matches)} matched files")

        for m in matches:
            if m not in seen:
                seen.add(m)
                resolved.append((m, override))

    if not resolved:
        raise ValueError(
            "no checkpoints resolved from --checkpoints / --checkpoints_file — "
            "nothing to evaluate."
        )
    return resolved

CMD_NAMES = {CMD_NEUTRAL: "neutral"}
try:
    from sim.tasks.joystick.play_env_cfg import CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT
    CMD_NAMES.update({CMD_UP: "up", CMD_DOWN: "down", CMD_LEFT: "left", CMD_RIGHT: "right",
                       CMD_HOME: "home"})
except ImportError:
    pass

NAME_TO_CMD = {v: k for k, v in CMD_NAMES.items() if k != CMD_HOME}
FULL_COMMAND_SET = [c for c in ALL_COMMANDS if c != CMD_HOME]


def names_to_commands(names_str: str):
    """'up,down' -> [CMD_UP, CMD_DOWN]; 'all' / 'full' -> every live command."""
    names_str = names_str.strip().lower()
    if names_str in ("all", "full"):
        return list(FULL_COMMAND_SET)
    names = [n.strip() for n in names_str.split(",") if n.strip()]
    unknown = [n for n in names if n not in NAME_TO_CMD]
    if unknown:
        raise ValueError(f"unrecognized command name(s) {unknown} — valid names: "
                          f"{list(NAME_TO_CMD.keys())} or 'all'")
    return [NAME_TO_CMD[n] for n in names]


def extract_job_name_from_path(ckpt_path: str):
    """Pulls '<job_name>' out of a path shaped like '.../LowLevel-<job_name>/...',
    matching this project's own checkpoint directory convention. Returns None
    if the path doesn't match that convention."""
    m = re.search(r"LowLevel-([^/\\]+)", ckpt_path)
    return m.group(1) if m else None


def parse_summary_configs(summary_path: str):
    """Parses ablation_summary.md's per-run config bullet line into a dict:
    job_name -> {"multi_task": bool, "single_cmd": str, "task_subset": str}.
    Returns {} if the file doesn't exist rather than raising, since summary
    lookup is only one of several fallback sources."""
    if not summary_path or not os.path.exists(summary_path):
        return {}

    configs = {}
    current_job = None
    job_re = re.compile(r"^\-\s+\*\*(.+?)\*\*")
    cfg_re = re.compile(
        r"camera=(\w+),\s*curriculum=(\w+),\s*multi_task=(\w+),\s*"
        r"single_cmd=(\S+),\s*task_subset=(\S+)"
    )
    with open(summary_path) as f:
        for line in f:
            jm = job_re.match(line.strip())
            if jm:
                current_job = jm.group(1)
                continue
            cm = cfg_re.search(line)
            if cm and current_job:
                _, _, multi_task, single_cmd, task_subset = cm.groups()
                configs[current_job] = {
                    "multi_task": multi_task.lower() == "true",
                    "single_cmd": single_cmd,
                    "task_subset": task_subset,
                }
                current_job = None
    return configs


RUN_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s/?#]+)")


def parse_summary_run_urls(summary_path: str):
    """Parses ablation_summary.md's '- run: <url>' lines into a dict:
    job_name -> {"entity": ..., "project": ..., "run_id": ...}. Jobs with no
    run: line (never actually trained, e.g. dependency-skipped) are simply
    absent from the returned dict, not an error."""
    if not summary_path or not os.path.exists(summary_path):
        return {}

    urls = {}
    current_job = None
    job_re = re.compile(r"^\-\s+\*\*(.+?)\*\*")
    run_re = re.compile(r"^\s*-\s+run:\s+(\S+)")
    with open(summary_path) as f:
        for line in f:
            jm = job_re.match(line.strip())
            if jm:
                current_job = jm.group(1)
                continue
            rm = run_re.match(line)
            if rm and current_job:
                m = RUN_URL_RE.search(rm.group(1))
                if m:
                    entity, project, run_id = m.groups()
                    urls[current_job] = {"entity": entity, "project": project, "run_id": run_id}
                current_job = None
    return urls


def download_run_history(entity: str, project: str, run_id: str, job_name: str, out_dir: str):
    """Downloads a run's FULL, unsampled training history via scan_history()
    (not .history(), which silently subsamples to 500 rows by default) and
    writes it to out_dir/<job_name>.csv. Returns the row count written, or
    None if the download failed (printed as a warning, not raised, so one
    bad job doesn't stop the rest of the batch)."""
    try:
        api = wandb.Api()
        # project names logged with spaces (e.g. "RL for Games") appear
        # URL-encoded in the summary's own links; decode back to the literal
        # name wandb.Api() expects.
        project_literal = project.replace("%20", " ")
        run = api.run(f"{entity}/{project_literal}/{run_id}")
        rows = list(run.scan_history())
        if not rows:
            print(f"    [warn] {job_name}: run {run_id} has no logged history rows")
            return None

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_name}.csv")
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"    downloaded {len(rows)} rows -> {out_path}")
        return len(rows)
    except Exception as e:
        print(f"    [warn] could not download history for {job_name} "
              f"({entity}/{project}/{run_id}): {type(e).__name__}: {e}")
        return None


def heuristic_commands_from_job_name(job_name: str):
    """Last-resort fallback: guesses active commands from the job name's own
    text, e.g. 'only_up_all_on' -> [up], 'left_up_all_on' -> [left, up],
    'all_all_on' -> full set. This is genuinely a guess — the job-naming
    convention has not been perfectly consistent in this project (job names
    and their actual trained command have mismatched before), so this is
    only used when every more reliable source has failed, and is always
    printed loudly rather than applied silently."""
    tokens = job_name.lower().split("_")
    found = [NAME_TO_CMD[t] for t in tokens if t in NAME_TO_CMD]
    if found:
        return found
    if "all" in tokens:
        return list(FULL_COMMAND_SET)
    return None


def resolve_active_commands(ckpt_path, raw_active_commands, override_str, summary_configs):
    """Priority order: checkpoint's own saved field > explicit override from
    the checkpoints file > ablation_summary.md lookup by job name > heuristic
    parse of the job name. The source actually used is always printed."""
    if raw_active_commands is not None:
        print(f"  active_commands source: checkpoint metadata -> {raw_active_commands}")
        return raw_active_commands

    if override_str is not None:
        cmds = names_to_commands(override_str)
        print(f"  active_commands source: checkpoints-file override ('{override_str}') -> {cmds}")
        return cmds

    job_name = extract_job_name_from_path(ckpt_path)
    if job_name and job_name in summary_configs:
        cfg = summary_configs[job_name]
        if cfg["multi_task"]:
            if cfg["task_subset"] and cfg["task_subset"] != "-":
                cmds = names_to_commands(cfg["task_subset"])
            else:
                cmds = list(FULL_COMMAND_SET)
        else:
            cmds = names_to_commands(cfg["single_cmd"])
        print(f"  active_commands source: ablation_summary.md (job_name='{job_name}') -> {cmds}")
        return cmds

    if job_name:
        guessed = heuristic_commands_from_job_name(job_name)
        if guessed is not None:
            print(f"  [WARNING] active_commands source: HEURISTIC guess from job name "
                  f"'{job_name}' -> {guessed} — verify this is correct, this is the least "
                  f"reliable source and this project's job names have been wrong before.")
            return guessed

    raise ValueError(
        f"{ckpt_path}: could not determine active_commands from checkpoint metadata, "
        f"checkpoints-file override, ablation_summary.md, or job-name heuristic. "
        f"Add an explicit override in the checkpoints file, e.g.:\n"
        f"    {ckpt_path}|up,down"
    )


def initialize_and_snapshot_home(base_env, device, n_steps=200, arrival_threshold=None):
    robot = base_env.scene["robot"]
    N = base_env.num_envs
    tol = arrival_threshold if arrival_threshold is not None else ARRIVAL_THRESHOLD
    target_t = torch.tensor(POSITIONS["home"], dtype=torch.float32, device=device).unsqueeze(0).repeat(N, 1)
    confirmed = torch.zeros(N, dtype=torch.int32, device=device)

    for _ in range(n_steps):
        robot.set_joint_position_target(target_t)
        robot.write_data_to_sim()
        base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())
        simulation_app.update()
        max_error = torch.abs(robot.data.joint_pos - target_t).max(dim=1).values
        confirmed = torch.where(max_error < tol, confirmed + 1, torch.zeros_like(confirmed))
        if (confirmed >= 5).all():
            break


def load_model(checkpoint_path: str, device: str, override_str, summary_configs):
    """Reads a checkpoint's own saved metadata and builds a matching Brain.
    If the checkpoint predates the active_commands save fix, falls back
    through override_str -> ablation_summary.md -> job-name heuristic."""
    raw = torch.load(checkpoint_path, map_location=device)

    use_camera = raw.get("use_camera", True)
    num_commands = raw.get("num_commands", 6)
    raw_active_commands = raw.get("active_commands", None)

    active_commands = resolve_active_commands(
        checkpoint_path, raw_active_commands, override_str, summary_configs
    )
    if len(active_commands) != num_commands:
        raise ValueError(
            f"{checkpoint_path}: resolved {len(active_commands)} active_commands "
            f"({active_commands}) but the checkpoint's actor was built with "
            f"num_commands={num_commands} — these must match. Check the override "
            f"or summary source used above."
        )

    brain = Brain(use_camera=use_camera, num_commands=num_commands)
    brain.load_checkpoint(checkpoint_path)
    brain.actor.eval()
    brain.critic.eval()

    return brain, use_camera, active_commands


def evaluate_model(base_env, device, brain, use_camera, active_commands, episodes_target):
    N = base_env.num_envs
    cmd_to_head_idx = {real_cmd: idx for idx, real_cmd in enumerate(active_commands)}
    is_single_task = len(active_commands) == 1

    base_env.cfg.episode_length_s = args_cli.eval_episode_length_s
    base_env.single_task_mode = is_single_task

    if is_single_task:
        base_env.command_manager.get_term("joystick_cmd")._command[:] = active_commands[0]
    else:
        base_env.command_manager.get_term("joystick_cmd").sampling_pool = active_commands

    obs, _ = base_env.reset() if hasattr(base_env, "reset") else (None, None)
    initialize_and_snapshot_home(base_env, device)

    if use_camera:
        frame_stacks = [Frames(n=3) for _ in range(N)]
        update_frame_stack(base_env, frame_stacks, reset_ids=list(range(N)))
        cam_states = np.stack([fs._get_state() for fs in frame_stacks])
    else:
        frame_stacks = None
        cam_states = None

    joint_states = base_env.scene["robot"].data.joint_pos.cpu().numpy()

    def read_commands():
        real = base_env.command_manager.get_command("joystick_cmd").cpu().numpy()
        if is_single_task:
            real = np.full((N,), active_commands[0], dtype=real.dtype)
        net = np.array([cmd_to_head_idx[int(c)] for c in real.tolist()], dtype=np.int64)
        return real, net

    commands_real, commands = read_commands()

    action_decision = brain.predict_next_action_batch(
        cam_states, joint_states, commands, deterministic=args_cli.deterministic
    )

    attempts = {c: 0 for c in active_commands}
    successes = {c: 0 for c in active_commands}
    episodes_done = 0
    decision_steps = np.zeros(N, dtype=int)
    pbar = tqdm(total=episodes_target, desc="  eval episodes", leave=False)

    while episodes_done < episodes_target:
        commands_real, commands = read_commands()
        current_joints = base_env.scene["robot"].data.joint_pos.cpu().numpy()
        target_joints = current_joints + action_decision

        obs, rewards, terminated, truncated, info = base_env.step(
            torch.tensor(target_joints, dtype=torch.float32, device=device)
        )
        dones = terminated | truncated
        decision_steps += 1

        reset_ids = torch.where(dones)[0].cpu().tolist()
        if use_camera:
            update_frame_stack(base_env, frame_stacks, reset_ids=reset_ids if reset_ids else None)
            cam_next = np.stack([fs._get_state() for fs in frame_stacks])
        else:
            cam_next = None
        joint_next = base_env.scene["robot"].data.joint_pos.cpu().numpy()

        episodes_before_this_step = episodes_done
        for i in range(N):
            if bool(dones[i].item()) and episodes_done < episodes_target:
                success_i = bool(terminated[i].item())
                real_cmd_i = int(commands_real[i])
                attempts[real_cmd_i] += 1
                if success_i:
                    successes[real_cmd_i] += 1
                episodes_done += 1
                decision_steps[i] = 0
        pbar.update(episodes_done - episodes_before_this_step)

        action_decision = brain.predict_next_action_batch(
            cam_next, joint_next, commands, deterministic=args_cli.deterministic
        )
        cam_states = cam_next
        joint_states = joint_next

    pbar.close()

    per_command_rate = {
        CMD_NAMES.get(c, str(c)): (successes[c] / attempts[c] if attempts[c] > 0 else float("nan"))
        for c in active_commands
    }
    rates = [r for r in per_command_rate.values() if not np.isnan(r)]
    total_attempts = sum(attempts.values())
    total_successes = sum(successes.values())

    return {
        "per_command_rate": per_command_rate,
        "per_command_attempts": {CMD_NAMES.get(c, str(c)): attempts[c] for c in active_commands},
        "min_rate": min(rates) if rates else float("nan"),
        "max_rate": max(rates) if rates else float("nan"),
        "mean_rate": float(np.mean(rates)) if rates else float("nan"),
        "overall_rate": (total_successes / total_attempts) if total_attempts > 0 else float("nan"),
        "total_episodes": total_attempts,
    }


def peek_use_camera(ckpt_path: str) -> bool:
    """Reads just the use_camera flag from a checkpoint's saved metadata,
    without constructing a Brain — used to partition checkpoints by camera
    requirement BEFORE any environment is built, so no-camera checkpoints
    can be evaluated against a camera-free (and therefore faster) env
    instance rather than a shared one that always renders regardless."""
    raw = torch.load(ckpt_path, map_location="cpu")
    return raw.get("use_camera", True)


def evaluate_checkpoint_group(base_env, device, group, summary_configs, group_label):
    """Runs the load -> evaluate -> print loop for one partition of the
    checkpoint list (all sharing the same use_camera requirement, since
    they're being evaluated against one already-built base_env). Returns a
    dict of model_name -> result, same shape as the full run's all_results."""
    results = {}
    if not group:
        return results

    print(f"\n{'#'*60}\n# Evaluating {group_label} group ({len(group)} checkpoint(s))\n{'#'*60}")

    for ckpt_path, override_str in group:
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'='*60}\nEvaluating: {model_name}  ({ckpt_path})\n{'='*60}")

        try:
            brain, use_camera, active_commands = load_model(
                ckpt_path, device, override_str, summary_configs
            )
        except Exception as e:
            print(f"  [SKIP] failed to load: {type(e).__name__}: {e}")
            continue

        print(f"  use_camera={use_camera}  active_commands={active_commands} "
              f"({[CMD_NAMES.get(c, str(c)) for c in active_commands]})")

        with torch.no_grad():
            result = evaluate_model(
                base_env, device, brain, use_camera, active_commands,
                episodes_target=args_cli.episodes_per_model,
            )

        results[model_name] = result

        print(f"  episodes evaluated: {result['total_episodes']}")
        for name, rate in result["per_command_rate"].items():
            n_attempts = result["per_command_attempts"][name]
            print(f"    {name:>8s}: {rate:.3f}  (n={n_attempts})")
        print(f"  min={result['min_rate']:.3f}  max={result['max_rate']:.3f}  "
              f"mean={result['mean_rate']:.3f}  overall={result['overall_rate']:.3f}")

    return results


def build_and_prime_env(task: str, device: str, num_envs: int, use_camera: bool):
    """Builds one Isaac Lab environment, optionally with the camera scene
    removed entirely (env_cfg.scene.side = None) before construction — this
    must happen before gym.make(), same as train.py's main(), since IsaacLab
    never allocates the camera's render resources if the field is None at
    scene-construction time, not just skipped afterward in Python."""
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if not use_camera:
        env_cfg.scene.side = None
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped

    env.reset()
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    return env, base_env


def main():
    if args_cli.checkpoints or args_cli.checkpoints_file:
        checkpoint_list = resolve_checkpoint_list(args_cli.checkpoints, args_cli.checkpoints_file)
        print(f"resolved {len(checkpoint_list)} checkpoint(s) from --checkpoints/--checkpoints_file:")
    elif args_cli.summary:
        checkpoint_list = resolve_checkpoints_from_summary(args_cli.summary)
        print(f"resolved {len(checkpoint_list)} checkpoint(s) directly from {args_cli.summary}:")
    else:
        raise ValueError("provide --checkpoints, --checkpoints_file, or --summary "
                          "(to derive checkpoints from the summary's own job names)")

    for path, override in checkpoint_list:
        print(f"  - {path}" + (f"   [override: {override}]" if override else ""))

    summary_configs = parse_summary_configs(args_cli.summary)
    if args_cli.summary:
        print(f"loaded {len(summary_configs)} job config(s) from {args_cli.summary}")

    if args_cli.download_history:
        if not args_cli.summary:
            print("[note] --download_history requested but --summary not given — "
                  "skipping history download (no run URLs available).")
        else:
            run_urls = parse_summary_run_urls(args_cli.summary)
            print(f"\n{'='*60}\nDownloading wandb training history -> {args_cli.history_out_dir}/\n{'='*60}")
            for path, _ in checkpoint_list:
                job_name = extract_job_name_from_path(path)
                if not job_name:
                    continue

                existing_path = os.path.join(args_cli.history_out_dir, f"{job_name}.csv")
                if os.path.exists(existing_path):
                    print(f"    [skip] {job_name}: {existing_path} already exists — "
                          f"delete it first if you want to force a re-download")
                    continue

                info = run_urls.get(job_name)
                if info is None:
                    print(f"    [skip] {job_name}: no run URL in {args_cli.summary} "
                          f"(job was likely dependency-skipped, never trained)")
                    continue
                print(f"  {job_name}:")
                download_run_history(info["entity"], info["project"], info["run_id"],
                                      job_name, args_cli.history_out_dir)

    global DECISION_STEPS_EVAL
    DECISION_STEPS_EVAL = args_cli.eval_decision_steps  # noqa: F841 (kept for clarity/logging use)

    # ── partition checkpoints by their own saved use_camera flag, BEFORE
    # any environment is built, so no-camera checkpoints never pay the cost
    # of a camera scene they don't use ────────────────────────────────────
    camera_group, no_camera_group = [], []
    for ckpt_path, override_str in checkpoint_list:
        try:
            uc = peek_use_camera(ckpt_path)
        except Exception as e:
            print(f"[warn] could not peek use_camera for {ckpt_path}: {e} — assuming True")
            uc = True
        (camera_group if uc else no_camera_group).append((ckpt_path, override_str))

    print(f"\npartitioned checkpoints: {len(camera_group)} with camera, "
          f"{len(no_camera_group)} without camera")

    all_results = {}

    if no_camera_group:
        env_nc, base_env_nc = build_and_prime_env(
            args_cli.task, args_cli.device, args_cli.num_envs, use_camera=False
        )
        all_results.update(evaluate_checkpoint_group(
            base_env_nc, str(base_env_nc.device), no_camera_group, summary_configs, "no-camera"
        ))
        env_nc.close()

    if camera_group:
        env_c, base_env_c = build_and_prime_env(
            args_cli.task, args_cli.device, args_cli.num_envs, use_camera=True
        )
        all_results.update(evaluate_checkpoint_group(
            base_env_c, str(base_env_c.device), camera_group, summary_configs, "camera"
        ))
        env_c.close()

    print(f"\n{'='*60}\nSUMMARY (all models)\n{'='*60}")
    print(f"{'model':<30s} {'min':>7s} {'max':>7s} {'mean':>7s} {'overall':>7s} {'episodes':>9s}")
    for name, r in all_results.items():
        print(f"{name:<30s} {r['min_rate']:>7.3f} {r['max_rate']:>7.3f} "
              f"{r['mean_rate']:>7.3f} {r['overall_rate']:>7.3f} {r['total_episodes']:>9d}")

    if args_cli.output_csv:
        with open(args_cli.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "command", "success_rate", "attempts",
                              "min_rate", "max_rate", "mean_rate", "overall_rate"])
            for name, r in all_results.items():
                for cmd_name, rate in r["per_command_rate"].items():
                    writer.writerow([name, cmd_name, rate, r["per_command_attempts"][cmd_name],
                                      r["min_rate"], r["max_rate"], r["mean_rate"], r["overall_rate"]])
        print(f"\nwrote per-model, per-command CSV -> {args_cli.output_csv}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)