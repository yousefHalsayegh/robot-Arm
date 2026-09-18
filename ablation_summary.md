# Ablation sweep summary

Started: 2026-09-18 19:36:24 UTC


## No camera or Curr

- **only_up_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5011 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=up, task_subset=-
- **only_left_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5004 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=left, task_subset=-
- **only_right_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5013 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=right, task_subset=-
- **only_down_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5003 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=down, task_subset=-
- **left_up_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5026 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5037 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down
- **all_no_camera_or_curr** — ALREADY COMPLETE (skipped, checkpoint at episode 5000 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=-

## No Camera

- **only_up_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5052 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=up, task_subset=-
- **only_left_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5012 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=left, task_subset=-
- **only_right_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5005 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=right, task_subset=-
- **only_down_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5047 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=down, task_subset=-
- **left_up_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5255 >= target 5000)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_no_camera** — ALREADY COMPLETE (skipped, checkpoint at episode 5151 >= target 5000)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down
- **all_no_camera** — completed
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=-, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/vgil7tlr
  - log: `ablation_logs/all_no_camera.log`

## No Curr

