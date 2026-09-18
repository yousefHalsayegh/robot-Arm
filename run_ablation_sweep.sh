set -uo pipefail  


TASK="${TASK:-Player}"                 # e.g. Isaac-Joystick-Play-v0
EPISODES="${EPISODES:-5000}"
FIXED_EPISODE_LENGTH_S="${FIXED_EPISODE_LENGTH_S:-10.0}"   # used only when curriculum is off

NUM_ENVS_CAMERA="${NUM_ENVS_CAMERA:-64}"
NUM_ENVS_NO_CAMERA="${NUM_ENVS_NO_CAMERA:-1024}"

SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.8}"

LOG_DIR="ablation_logs"
SUMMARY_MD="ablation_summary.md"

mkdir -p "$LOG_DIR"
: > "$SUMMARY_MD"   # truncate/create fresh

echo "# Ablation sweep summary" >> "$SUMMARY_MD"
echo "" >> "$SUMMARY_MD"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$SUMMARY_MD"
echo "" >> "$SUMMARY_MD"


CONFIGS=(
  # job_name  use_camera  use_curr  multi_task  single_cmd  task_subset  prereqs  group_label

  # ---- No camera or Curr ----
  "only_up_no_camera_or_curr    false false false up    -       -                                                                                    No camera or Curr"
  "only_left_no_camera_or_curr  false false false left  -       -                                                                                    No camera or Curr"
  "only_right_no_camera_or_curr false false false right -       -                                                                                    No camera or Curr"
  "only_down_no_camera_or_curr  false false false down  -       -                                                                                    No camera or Curr"
  "left_up_no_camera_or_curr    false false true  -     left,up only_left_no_camera_or_curr,only_up_no_camera_or_curr                                 No camera or Curr"
  "up_down_no_camera_or_curr    false false true  -     up,down only_up_no_camera_or_curr,only_down_no_camera_or_curr                                 No camera or Curr"
  "all_no_camera_or_curr        false false true  -     -       only_up_no_camera_or_curr,only_left_no_camera_or_curr,only_right_no_camera_or_curr,only_down_no_camera_or_curr  No camera or Curr"

  # ---- No Camera (curriculum ON) ----
  "only_up_no_camera    false true false up    -       -                                              No Camera"
  "only_left_no_camera  false true false left  -       -                                              No Camera"
  "only_right_no_camera false true false right -       -                                              No Camera"
  "only_down_no_camera  false true false down  -       -                                              No Camera"
  "left_up_no_camera    false true true  -     left,up only_left_no_camera,only_up_no_camera            No Camera"
  "up_down_no_camera    false true true  -     up,down only_up_no_camera,only_down_no_camera            No Camera"
  "all_no_camera        false true true  -     -       only_up_no_camera,only_left_no_camera,only_right_no_camera,only_down_no_camera  No Camera"

  # ---- No Curr (camera ON) ----
  "only_up_no_curr    true false false up    -       -                                No Curr"
  "only_left_no_curr  true false false left  -       -                                No Curr"
  "left_up_no_curr    true false true  -     left,up only_left_no_curr,only_up_no_curr  No Curr"
  "up_down_no_curr    true false true  -     up,down only_up_no_curr                    No Curr"
  "all_no_curr         true false true  -     -       only_up_no_curr,only_left_no_curr  No Curr"

  # ---- All on (camera + curriculum) ----
  "only_up_all_on    true true false up    -       -                                All on"
  "only_left_all_on  true true false left  -       -                                All on"
  "all_all_on         true true true  -     -       only_up_all_on,only_left_all_on  All on"

  # ---- Task subset (multi-task, restricted command set) ----
  "left_up_all_on   true true true  -     left,up only_left_all_on,only_up_all_on  Task subset"
  "up_down_all_on   true true true  -     up,down only_up_all_on                    Task subset"
)

CURRENT_GROUP=""
declare -A SUCCESS_RATE   # job_name -> final success_rate captured from wandb (bash 4+ associative array)

for entry in "${CONFIGS[@]}"; do
  read -r JOB_NAME USE_CAMERA USE_CURR MULTI_TASK SINGLE_CMD TASK_SUBSET PREREQS GROUP_LABEL <<< "$entry"

  if [[ "$GROUP_LABEL" != "$CURRENT_GROUP" ]]; then
    CURRENT_GROUP="$GROUP_LABEL"
    echo "" >> "$SUMMARY_MD"
    echo "## $GROUP_LABEL" >> "$SUMMARY_MD"
    echo "" >> "$SUMMARY_MD"
    echo "----------------------------------------"
    echo "GROUP: $GROUP_LABEL"
    echo "----------------------------------------"
  fi

  
  SKIP_REASON=""
  if [[ "$PREREQS" != "-" ]]; then
    IFS=',' read -ra PREREQ_LIST <<< "$PREREQS"
    for prereq in "${PREREQ_LIST[@]}"; do
      if [[ -n "${SUCCESS_RATE[$prereq]+x}" ]]; then
        rate="${SUCCESS_RATE[$prereq]}"
        below=$(awk -v r="$rate" -v t="$SUCCESS_THRESHOLD" 'BEGIN{print (r < t) ? 1 : 0}')
        if [[ "$below" -eq 1 ]]; then
          SKIP_REASON="prerequisite '$prereq' success_rate=$rate < threshold=$SUCCESS_THRESHOLD"
          break
        fi
      fi
    done
  fi

  if [[ -n "$SKIP_REASON" ]]; then
    echo ""
    echo ">>> Skipping: $JOB_NAME  ($SKIP_REASON)"
    {
      echo "- **${JOB_NAME}** — SKIPPED"
      echo "  - reason: ${SKIP_REASON}"
      echo "  - camera=${USE_CAMERA}, curriculum=${USE_CURR}, multi_task=${MULTI_TASK}, single_cmd=${SINGLE_CMD}, task_subset=${TASK_SUBSET}"
    } >> "$SUMMARY_MD"
    continue
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

  FULL_DIR="runs/LowLevel-${JOB_NAME}/Full"
  CKPT_DIR="runs/LowLevel-${JOB_NAME}/Checkpoints"
  RESUME_FLAG=""
  RESUME_WANDB_ID=""
  RESUME_WANDB_ENTITY=""

  # ── find the highest-episode checkpoint across BOTH Full/ and
  # Checkpoints/ ─────────────────────────────────────────────────────────
  # A crash or interrupt now ALSO writes a checkpoint into Full/ (so its
  # mere existence no longer means training reached the full episode
  # target) — the genuinely furthest-along save could be sitting in either
  # directory, so both are scanned and compared directly against EPISODES.
  HIGHEST_EP=0
  HIGHEST_PATH=""
  for dir in "$FULL_DIR" "$CKPT_DIR"; do
    if compgen -G "${dir}/manipulation_brain_*.pth" > /dev/null 2>&1; then
      for f in "${dir}"/manipulation_brain_*.pth; do
        ep=$(basename "$f" | sed -E 's/manipulation_brain_([0-9]+)\.pth/\1/')
        if [[ "$ep" =~ ^[0-9]+$ ]] && (( ep > HIGHEST_EP )); then
          HIGHEST_EP=$ep
          HIGHEST_PATH="$f"
        fi
      done
    fi
  done

  # A checkpoint can only be trusted if it was actually saved by a matching
  # config — a leftover/misplaced checkpoint with the wrong use_camera
  # would otherwise either be silently skipped as "already complete" or
  # cause train.py's own architecture check to hard-fail on resume. Peek
  # the checkpoint's saved flag directly and discard it if it disagrees
  # with this job's own definition.
  if (( HIGHEST_EP > 0 )); then
    CKPT_USE_CAMERA=$(python3 -c "
import torch, sys
d = torch.load(sys.argv[1], map_location='cpu')
print('true' if d.get('use_camera', True) else 'false')
" "$HIGHEST_PATH" 2>/dev/null)

    if [[ "$CKPT_USE_CAMERA" != "$USE_CAMERA" ]]; then
      echo "    [warn] checkpoint at $HIGHEST_PATH has use_camera=$CKPT_USE_CAMERA but "
      echo "           job '$JOB_NAME' requires use_camera=$USE_CAMERA — this checkpoint "
      echo "           does not belong to this job's config; ignoring it and starting fresh"
      HIGHEST_EP=0
      HIGHEST_PATH=""
    fi
  fi

  if (( HIGHEST_EP >= EPISODES )); then
    echo ""
    echo ">>> Skipping: $JOB_NAME  (already completed — checkpoint at episode $HIGHEST_EP >= target $EPISODES)"
    {
      echo "- **${JOB_NAME}** — ALREADY COMPLETE (skipped, checkpoint at episode ${HIGHEST_EP} >= target ${EPISODES})"
      echo "  - camera=${USE_CAMERA}, curriculum=${USE_CURR}, multi_task=${MULTI_TASK}, single_cmd=${SINGLE_CMD}, task_subset=${TASK_SUBSET}"
    } >> "$SUMMARY_MD"
    continue
  elif (( HIGHEST_EP > 0 )); then
    # train.py's own checkpoint loader only ever looks inside Checkpoints/
    # (ckpt_path = f"{ckpt_dir}/manipulation_brain_{args.checkpoint}.pth",
    # ckpt_dir hardcoded to .../Checkpoints) — if the furthest-along save
    # is actually sitting in Full/ (written by a crash/interrupt rather
    # than a genuine mid-training mid_save), copy it into Checkpoints/
    # under the same filename so that lookup finds it.
    if [[ "$HIGHEST_PATH" == "${FULL_DIR}/"* ]]; then
      mkdir -p "$CKPT_DIR"
      cp "$HIGHEST_PATH" "${CKPT_DIR}/manipulation_brain_${HIGHEST_EP}.pth"
      echo "    incomplete run (episode $HIGHEST_EP of $EPISODES) — copied its Full checkpoint into Checkpoints/ for resuming"
    fi
    RESUME_FLAG="-chk=$HIGHEST_EP"
    echo "    resuming from checkpoint at episode $HIGHEST_EP (target: $EPISODES)"

    if [[ -f "$LOG_FILE" ]]; then
      WANDB_URL_PREV=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[^ ]+' "$LOG_FILE" | head -n 1)
      RESUME_WANDB_ID=$(echo "$WANDB_URL_PREV" | sed -E 's|.*/runs/([^/?#]+).*|\1|')
      RESUME_WANDB_ENTITY=$(echo "$WANDB_URL_PREV" | sed -E 's|https://wandb\.ai/([^/]+)/.*|\1|')
      if [[ -n "$RESUME_WANDB_ID" ]]; then
        echo "    resuming wandb run: $RESUME_WANDB_ENTITY/$RESUME_WANDB_ID"
      fi
    fi
  fi

  echo ""
  echo ">>> Running: $JOB_NAME"
  echo "    num_envs=$RUN_NUM_ENVS  $CAM_FLAG $CURR_FLAG $TASK_FLAG $CMD_FLAG $RESUME_FLAG"

  train \
    --task="$TASK" \
    --enable_cameras \
    --headless \
    --num_envs="$RUN_NUM_ENVS" \
    -ep="$EPISODES" \
    -jn="$JOB_NAME" \
    --fixed_episode_length_s "$FIXED_EPISODE_LENGTH_S" \
    --wandb_resume_id "$RESUME_WANDB_ID" \
    --wandb_entity "$RESUME_WANDB_ENTITY" \
    $CAM_FLAG $CURR_FLAG $TASK_FLAG $CMD_FLAG $RESUME_FLAG \
    2>&1 | tee -a "$LOG_FILE"

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

  
  if [[ "$SINGLE_CMD" != "-" ]]; then
    metric_value=$(grep "^FINAL_SUCCESS_RATE|${SINGLE_CMD}|" "$LOG_FILE" | tail -n 1 | awk -F'|' '{print $3}')

    if [[ -n "$metric_value" && "$metric_value" != "nan" ]]; then
      SUCCESS_RATE["$JOB_NAME"]="$metric_value"
      echo "    captured success_rate/${SINGLE_CMD} = $metric_value  (for future prereq checks)"
    else
      echo "    [note] no FINAL_SUCCESS_RATE line found for '${SINGLE_CMD}' in $LOG_FILE — "
      echo "           any later run depending on $JOB_NAME will treat this as inconclusive"
    fi
  fi

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