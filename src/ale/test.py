"""
benchmark_envs.py

Quick throughput sweep across different --environment values, measuring
REAL steps/sec (not just memory, which the existing --dry_run already
covers) — this is the missing piece needed to actually pick num_envs,
since memory being fine says nothing about whether more envs step faster
on this specific CPU.

Also benchmarks Brain.train() update cost separately, so you can combine
both numbers to reason about overall iteration time (env-stepping +
updates) at a given (num_envs, updates) pair, rather than just picking
num_envs off the env-stepping benchmark alone.

Usage:
    python benchmark_envs.py --env_counts 16,32,64,128,256,512 --step_iters 300
    python benchmark_envs.py --env_counts 16,32,64 --bench_updates --updates_list 4,8,16,32
"""

import argparse
import time

import numpy as np
import torch
from ale_py import AtariVectorEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env_counts", type=str, default="16,32,64,128,256",
                     help="comma-separated list of num_envs values to test")
parser.add_argument("--step_iters", type=int, default=3000,
                     help="number of env.step() calls to time per env_count")
parser.add_argument("--warmup_iters", type=int, default=20,
                     help="untimed steps before the timed window, lets threads/caches settle")
parser.add_argument("--game", type=str, default="pong")
parser.add_argument("--bench_updates", action="store_true",
                     help="also benchmark Brain.train() cost — requires game_rl.py's "
                          "Brain class to be importable from the current directory")
parser.add_argument("--updates_list", type=str, default="4,8,16,32,64",
                     help="comma-separated update counts to time, if --bench_updates")
parser.add_argument("--batch", type=int, default=256)
args = parser.parse_args()

ENV_COUNTS = [int(x) for x in args.env_counts.split(",")]


def bench_env_stepping(num_envs: int, game: str, step_iters: int, warmup_iters: int):
    env = AtariVectorEnv(
        game=game, num_envs=num_envs, frameskip=4, grayscale=True,
        stack_num=4, img_height=84, img_width=84, noop_max=30,
    )
    env.reset(seed=42)
    action_space_n = env.single_action_space.n

    # untimed warmup — first few steps often pay one-off thread-pool/cache
    # setup cost that would otherwise bias the timed window
    for _ in range(warmup_iters):
        actions = np.random.randint(0, action_space_n, size=num_envs)
        env.step(actions)

    t0 = time.time()
    for _ in range(step_iters):
        actions = np.random.randint(0, action_space_n, size=num_envs)
        env.step(actions)
    elapsed = time.time() - t0

    env.close()
    total_env_steps = step_iters * num_envs
    return total_env_steps / elapsed, elapsed


def bench_updates(brain, updates_list, warmup_updates=5):
    results = {}
    for n_updates in updates_list:
        for _ in range(warmup_updates):
            brain.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_updates):
            brain.train()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        results[n_updates] = (n_updates / elapsed, elapsed)
    return results


def main():
    print(f"{'='*70}\nBenchmark 1: environment-stepping throughput (CPU-bound)\n{'='*70}")
    print(f"{'num_envs':>10s} {'steps/sec':>14s} {'elapsed_s':>12s} {'notes':>30s}")

    step_results = {}
    prev_rate = None
    for n in ENV_COUNTS:
        rate, elapsed = bench_env_stepping(n, args.game, args.step_iters, args.warmup_iters)
        step_results[n] = rate
        note = ""
        if prev_rate is not None:
            scaling = rate / prev_rate
            note = f"{scaling:.2f}x prior rate"
        print(f"{n:>10d} {rate:>14.0f} {elapsed:>12.2f} {note:>30s}")
        prev_rate = rate

    best_n = max(step_results, key=step_results.get)
    print(f"\nBest env-stepping throughput: num_envs={best_n} ({step_results[best_n]:.0f} steps/sec)")
    print("If throughput plateaus or drops past some N, that N is roughly your "
          "CPU's real parallelism ceiling — going further just adds scheduling "
          "overhead without giving you more actual speed.")

    if args.bench_updates:
        print(f"\n{'='*70}\nBenchmark 2: Brain.train() update cost (GPU-bound)\n{'='*70}")
        try:
            from game_rl import Brain, config
        except ImportError as e:
            print(f"[skip] could not import Brain from game_rl.py: {e}")
            print("       run this script from the same directory as game_rl.py, "
                  "or adjust the import above to match your module layout.")
            return

        brain = Brain(
            lr=3e-4, wp=1000, b=args.batch, g=0.99, tau=0.005,
            ee=0.05, es=1.0, ed=500000, c=500000,
            d=True, tu="soft", tup=8000, n=True,
        )
        # fill the buffer with dummy transitions so train() doesn't just
        # return early on the warmup check
        dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)
        for _ in range(max(args.batch, 2000)):
            brain.buffer.push(dummy_state, 0, 0.0, dummy_state, False, 1)

        updates_list = [int(x) for x in args.updates_list.split(",")]
        update_results = bench_updates(brain, updates_list)

        print(f"{'updates':>10s} {'updates/sec':>14s} {'elapsed_s':>12s}")
        for n_updates, (rate, elapsed) in update_results.items():
            print(f"{n_updates:>10d} {rate:>14.1f} {elapsed:>12.3f}")

        print("\nCombine both tables: for a candidate (num_envs, updates) pair using "
              "updates = 0.25 * num_envs, estimated real time per training iteration is "
              "roughly (num_envs / env_steps_per_sec) + (updates / updates_per_sec). "
              "Pick whichever num_envs minimizes that sum, not whichever maximizes "
              "env-stepping throughput alone.")


if __name__ == "__main__":
    main()