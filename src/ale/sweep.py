import argparse
import subprocess
import sys
import time
import wandb
import os
import signal
import yaml

import psutil

def kill_process_tree(pid, timeout=3):
    """Kill a process and all of its descendants, regardless of whether
    they've detached into their own session/process group."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    procs = children + [parent]

    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()   # SIGKILL anything that ignored terminate()
        except psutil.NoSuchProcess:
            pass

def main(args):
    os.makedirs(args.logdir, exist_ok=True)

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Reusing existing sweep: {sweep_id}")
    else:
        with open(args.config, "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")

    full_sweep_path = f"{args.entity}/{args.project}/{sweep_id}" if args.entity else f"{args.project}/{sweep_id}"

    processes = []
    try:
        for i in range(args.agents):
            log_path = os.path.join(args.logdir, f"agent_{i}.log")
            log_file = open(log_path, "w")
            cmd = ["wandb", "agent", full_sweep_path]
            print(f"Launching agent {i} -> {log_path}")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((p, log_file))
            time.sleep(2)

        print(f"\n{args.agents} agents running against sweep {full_sweep_path}")
        print("Press Ctrl+C to stop all agents.\n")

        while True:
            time.sleep(10)
            alive = [p for p, _ in processes if p.poll() is None]
            if not alive:
                print("All agents finished.")
                break

    except KeyboardInterrupt:
        print("\nStopping all agents...")
        for p, log_file in processes:
            if p.poll() is None:
                kill_process_tree(p.pid)
            log_file.close()
        for p, _ in processes:
            p.wait()
        print("Done.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser("Launch a wandb sweep across N parallel agents")
    parser.add_argument("-c", "--config", type=str, default="sweep_config.yaml",
        help="Path to the sweep config YAML")
    parser.add_argument("-n", "--agents", type=int, default=10,
        help="Number of concurrent wandb agent processes to launch")
    parser.add_argument("-p", "--project", type=str, default="RL for Games",
        help="wandb project name (must match sweep_config.yaml's project, if set there)")
    parser.add_argument("-e", "--entity", type=str, default=None,
        help="wandb entity/team (optional, uses your default if omitted)")
    parser.add_argument("-l", "--logdir", type=str, default="logs",
        help="Directory to write each agent's stdout/stderr")
    parser.add_argument("--sweep_id", type=str, default=None,
        help="Reuse an existing sweep ID instead of creating a new one "
             "(useful to add more agents to a sweep already in progress)")
    main(parser.parse_args())