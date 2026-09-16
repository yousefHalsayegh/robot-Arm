#!/usr/bin/env bash
#
# Runs the full camera x curriculum x task ablation sweep (4 groups x 5 configs
# = 20 runs), logs each run's wandb URL as it appears, and writes a summary
# outline at the end.
#
# Fill in TASK / EPISODES / NUM_ENVS for your setup before running.

set -uo pipefail   # NOT `set -e` — a single failed run must not kill the sweep

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings — same across every run in the sweep, so the only variables
# are camera / curriculum / task, matching the ablation ladder's intent.
# ─────────────────────────────────────────────────────────────────────────────
TASK="${TASK:-Player}"                 # e.g. Isaac-Joystick-Play-v0
EPISODES="${EPISODES:-2500}"
FIXED_EPISODE_LENGTH_S="${FIXED_EPISODE_LENGTH_S:-10.0}"   # used only when curriculum is off

# Camera runs need a much smaller env count than no-camera runs (per today's
# VRAM/render-budget findings) — override either at invocation time, e.g.:
#   NUM_ENVS_CAMERA=16 NUM_ENVS_NO_CAMERA=128 ./run_ablation_sweep.sh
NUM_ENVS_CAMERA="${NUM_ENVS_CAMERA:-16}"
NUM_ENVS_NO_CAMERA="${NUM_ENVS_NO_CAMERA:-1024}"

LOG_DIR="ablation_logs"
SUMMARY_MD="ablation_summary.md"

mkdir -p "$LOG_DIR"
: > "$SUMMARY_MD"   # truncate/create fresh

echo "# Ablation sweep summary" >> "$SUMMARY_MD"
echo "" >> "$SUMMARY_MD"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$SUMMARY_MD"
echo "" >> "$SUMMARY_MD"

# ─────────────────────────────────────────────────────────────────────────────
# Config table: job_name  use_camera  use_curr  multi_task  single_cmd  task_subset  group_label
# single_cmd is "-" when multi_task=true (not used in that case).
# task_subset is "-" for the full command set, or a comma-separated list
# (no spaces) like "left,up" to train on only that subset — only meaningful
# when multi_task=true.
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS=(
  # ---- No camera or Curr ----
  "only_up_no_curr    false false false up    -       No camera or Curr"
  "only_left_no_curr  false false false left  -       No camera or Curr"
  "only_right_no_curr    false false false right    -       No camera or Curr"
  "only_down_no_curr  false false false down  -       No camera or Curr"
  "left_up_no_camera_or_curr    false false true  -     left,up  No camera or Curr"
  "up_down_no_camera_or_curr    false false true  -     up,down  No camera or Curr"
  "all_no_camera_or_curr    false false true  -     - No camera or Curr"
  

  # ---- No Camera (curriculum ON) ----
  "only_up_no_curr    false false false up    -       No camera or Curr"
  "only_left_no_curr  false false false left  -       No camera or Curr"
  "only_right_no_curr    false false false right    -       No camera or Curr"
  "only_down_no_curr  false false false down  -       No camera or Curr"
  "left_up_no_camera    false true true  -     left,up  No Camera"
  "up_down_no_camera    false true true  -     up,down  No Camera"
  "all_no_camera_or_curr    false false true  -     - No camera or Curr"

  # ---- No Curr (camera ON) ----
  "only_up_no_curr    true false false up    -       No Curr"
  "only_left_no_curr  true false false left  -       No Curr"
  "left_up_no_camera    true false true  -     left,up  No Curr"
  "up_down_no_camera    true false true  -     up,down  No Curr"
  "all_no_curr         true false true  -     -       No Curr"

  # ---- All on (camera + curriculum) ----
  "only_up_all_on    true true false up    -       All on"
  "only_left_all_on  true true false left  -       All on"
  "all_all_on         true true true  -     -       All on"
  "left_up_all_on   true true true  -     left,up  Task subset"
  "up_down_all_on   true true true  -     up,down  Task subset"


)

CURRENT_GROUP=""

for entry in "${CONFIGS[@]}"; do
  read -r JOB_NAME USE_CAMERA USE_CURR MULTI_TASK SINGLE_CMD TASK_SUBSET GROUP_LABEL <<< "$entry"

  if [[ "$GROUP_LABEL" != "$CURRENT_GROUP" ]]; then
    CURRENT_GROUP="$GROUP_LABEL"
    echo "" >> "$SUMMARY_MD"
    echo "## $GROUP_LABEL" >> "$SUMMARY_MD"
    echo "" >> "$SUMMARY_MD"
    echo "----------------------------------------"
    echo "GROUP: $GROUP_LABEL"
    echo "----------------------------------------"
  fi

  CAM_FLAG="--use_camera";            [[ "$USE_CAMERA" == "false" ]] && CAM_FLAG="--no-use_camera"
  CURR_FLAG="--use_episode_curriculum"; [[ "$USE_CURR" == "false" ]] && CURR_FLAG="--no-use_episode_curriculum"
  TASK_FLAG="--multi_task";           [[ "$MULTI_TASK" == "false" ]] && TASK_FLAG="--no-multi_task"

  if [[ "$USE_CAMERA" == "true" ]]; then
    RUN_NUM_ENVS="$NUM_ENVS_CAMERA"
  else
    RUN_NUM_ENVS="$NUM_ENVS_NO_CAMERA"
  fi

  CMD_FLAG=""
  if [[ "$MULTI_TASK" == "false" ]]; then
    CMD_FLAG="--single_task_command $SINGLE_CMD"
  elif [[ "$TASK_SUBSET" != "-" ]]; then
    CMD_FLAG="--task_subset $TASK_SUBSET"
  fi

  LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

  echo ""
  echo ">>> Running: $JOB_NAME"
  echo "    num_envs=$RUN_NUM_ENVS  $CAM_FLAG $CURR_FLAG $TASK_FLAG $CMD_FLAG"

  train \
    --task="$TASK" \
    --enable_cameras \
    --headless \
    --num_envs="$RUN_NUM_ENVS" \
    -ep="$EPISODES" \
    -jn="$JOB_NAME" \
    --fixed_episode_length_s "$FIXED_EPISODE_LENGTH_S" \
    $CAM_FLAG $CURR_FLAG $TASK_FLAG $CMD_FLAG \
    2>&1 | tee "$LOG_FILE"

  RUN_EXIT=${PIPESTATUS[0]}

  # wandb prints its run URL to stdout/stderr as it starts, e.g.:
  #   wandb: 🚀 View run at https://wandb.ai/<entity>/<project>/runs/<id>
  RUN_URL=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[^ ]+' "$LOG_FILE" | head -n 1)

  if [[ "$RUN_EXIT" -ne 0 ]]; then
    STATUS="FAILED (exit $RUN_EXIT)"
  else
    STATUS="completed"
  fi

  if [[ -z "$RUN_URL" ]]; then
    RUN_URL="(no wandb URL found in log — check $LOG_FILE)"
  fi

  echo "    status: $STATUS"
  echo "    url:    $RUN_URL"

  {
    echo "- **${JOB_NAME}** — ${STATUS}"
    echo "  - camera=${USE_CAMERA}, curriculum=${USE_CURR}, multi_task=${MULTI_TASK}, single_cmd=${SINGLE_CMD}, task_subset=${TASK_SUBSET}, num_envs=${RUN_NUM_ENVS}"
    echo "  - run: ${RUN_URL}"
    echo "  - log: \`${LOG_FILE}\`"
  } >> "$SUMMARY_MD"
done

echo "" >> "$SUMMARY_MD"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$SUMMARY_MD"

echo ""
echo "========================================"
echo "Sweep complete. Summary written to: $SUMMARY_MD"
echo "Per-run logs in: $LOG_DIR/"
echo "========================================"
cat "$SUMMARY_MD"