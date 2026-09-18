#!/usr/bin/env bash
#
# Launches the Rainbow DQN wandb sweep (sweep_config.yaml) via the existing
# Python multi-agent launcher, with defaults sized for running ALONGSIDE the
# Isaac Lab SAC sweep on the same machine (see today's memory/CPU/GPU
# discussion — this is the lower-priority workload, SAC is not).
#
# This is a thin wrapper: the Python launcher already correctly handles
# sweep creation, multi-agent dispatch, and clean process-tree shutdown on
# Ctrl+C — this script only adds priority control and resource monitoring
# around it, matching run_ablation_sweep.sh's structure/style.

set -uo pipefail

CONFIG="${CONFIG:-src/ale/sweep_config.yaml}"
AGENTS="${AGENTS:-5}"                       # matches the earlier "2 agents" time-constrained decision
PROJECT="${PROJECT:-RL for Games}"
ENTITY="${ENTITY:-}"                        # leave empty to use your wandb default entity
LOGDIR="${LOGDIR:-logs}"
LAUNCHER_SCRIPT="${LAUNCHER_SCRIPT:-src/ale/launch_sweep.py}"   # the Python script you pasted — rename here if it lives elsewhere

# Deprioritise relative to the SAC/Isaac Lab sweep, which is the priority
# workload on this machine right now. Positive nice values need no special
# privileges. Child processes (wandb agent -> sweep_train.py -> game_rl.py)
# inherit this automatically at fork time — no extra `nice` calls needed
# inside the Python launcher itself.
NICE_LEVEL="${NICE_LEVEL:-15}"

RESOURCE_WATCH_LOG="${RESOURCE_WATCH_LOG:-rainbow_sweep_resource_watch.log}"
RESOURCE_WATCH_INTERVAL_S="${RESOURCE_WATCH_INTERVAL_S:-120}"

# Optional: defer launch until the SAC sweep's own summary file shows its
# final "Finished:" line — written as the literal last action of
# run_ablation_sweep.sh, once ALL jobs (not just the currently-running one)
# have completed. This is deliberately NOT process-pattern matching (e.g.
# pgrep -f) — a per-job subprocess like `tee -a ablation_logs/<job>.log`
# only lives for the duration of ONE job and can create a false "sweep is
# done" reading in the gap between two jobs, which is exactly the race
# this needs to avoid. Empty (default) means launch immediately.
WAIT_FOR_SUMMARY="${WAIT_FOR_SUMMARY:-ablation_summary.md}"
WAIT_POLL_INTERVAL_S="${WAIT_POLL_INTERVAL_S:-120}"

mkdir -p "$LOGDIR"

if [[ -n "$WAIT_FOR_SUMMARY" ]]; then
  echo "Waiting for '$WAIT_FOR_SUMMARY' to show a 'Finished:' line "
  echo "(i.e. the SAC sweep to fully complete, not just its current job)..."
  waited_s=0
  while ! grep -q "^Finished:" "$WAIT_FOR_SUMMARY" 2>/dev/null; do
    sleep "$WAIT_POLL_INTERVAL_S"
    waited_s=$((waited_s + WAIT_POLL_INTERVAL_S))
    echo "  still waiting... (${waited_s}s elapsed, no 'Finished:' line yet)"
  done
  echo "'$WAIT_FOR_SUMMARY' shows completion — proceeding with launch."
  echo ""
fi

echo "========================================"
echo "Rainbow DQN sweep launch"
echo "========================================"
echo "config:        $CONFIG"
echo "agents:        $AGENTS"
echo "project:       $PROJECT"
echo "entity:        ${ENTITY:-<default>}"
echo "logdir:        $LOGDIR"
echo "nice level:    $NICE_LEVEL  (lower priority than the SAC sweep, deliberately)"
echo "resource log:  $RESOURCE_WATCH_LOG  (checked every ${RESOURCE_WATCH_INTERVAL_S}s)"
echo "========================================"
echo ""
echo "Reminder: capacity/updates are fixed inside $CONFIG itself — check that file's"
echo "values match the current situation (shared with SAC vs. running alone) before"
echo "trusting this launch, rather than assuming which config you're pointing at."
echo ""

# ── background resource monitor ──────────────────────────────────────────
# Runs independently of the sweep itself; check swap and GPU memory trend
# over time here rather than needing to babysit the terminal.
(
  while true; do
    {
      echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
      free -h
      if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
      fi
      echo ""
    } >> "$RESOURCE_WATCH_LOG"
    sleep "$RESOURCE_WATCH_INTERVAL_S"
  done
) &
MONITOR_PID=$!

cleanup() {
  echo ""
  echo "Stopping resource monitor (pid $MONITOR_PID)..."
  kill "$MONITOR_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ── launch the sweep ──────────────────────────────────────────────────────
# Running in the foreground (not backgrounded) so Ctrl+C here propagates
# directly to the Python launcher, which already handles it correctly
# (kill_process_tree on every agent, including detached wandb-agent
# subprocesses started with start_new_session=True).
nice -n "$NICE_LEVEL" python "$LAUNCHER_SCRIPT" \
  --config "$CONFIG" \
  --agents "$AGENTS" \
  --project "$PROJECT" \
  ${ENTITY:+--entity "$ENTITY"} \
  --logdir "$LOGDIR"

echo ""
echo "========================================"
echo "Rainbow sweep launcher exited."
echo "Per-agent logs: ${LOGDIR}/agent_*.log"
echo "Resource watch: ${RESOURCE_WATCH_LOG}"
echo "========================================"