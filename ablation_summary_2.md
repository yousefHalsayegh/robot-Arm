# Ablation sweep summary

Started: 2026-09-17 22:00:00 UTC


## No camera or Curr

- **only_up_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=false, single_cmd=up, task_subset=-
- **only_left_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=false, single_cmd=left, task_subset=-
- **only_right_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=false, single_cmd=right, task_subset=-
- **only_down_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=false, single_cmd=down, task_subset=-
- **left_up_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down
- **all_no_camera_or_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=-

## No Camera

- **only_up_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=false, single_cmd=up, task_subset=-
- **only_left_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=false, single_cmd=left, task_subset=-
- **only_right_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=false, single_cmd=right, task_subset=-
- **only_down_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=false, single_cmd=down, task_subset=-
- **left_up_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down
- **all_no_camera** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=-

## No Curr

- **only_up_no_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=true, curriculum=false, multi_task=false, single_cmd=up, task_subset=-
- **only_left_no_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=true, curriculum=false, multi_task=false, single_cmd=left, task_subset=-
- **left_up_no_curr** — ALREADY COMPLETE (skipped, Full checkpoint found)
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_no_curr** — completed
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/9utw2393
  - log: `ablation_logs/up_down_no_curr.log`
- **all_no_curr** — completed
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/aj3smfe6
  - log: `ablation_logs/all_no_curr.log`

## All on

- **only_up_all_on** — completed
  - camera=true, curriculum=true, multi_task=false, single_cmd=up, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/9bfc5aoo
  - log: `ablation_logs/only_up_all_on.log`
- **only_left_all_on** — completed
  - camera=true, curriculum=true, multi_task=false, single_cmd=left, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/97hydrx1
  - log: `ablation_logs/only_left_all_on.log`
- **all_all_on** — SKIPPED
  - reason: prerequisite 'only_up_all_on' success_rate=0.25 < threshold=0.8
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=-

## Task subset

- **left_up_all_on** — SKIPPED
  - reason: prerequisite 'only_left_all_on' success_rate=0.4 < threshold=0.8
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up
- **up_down_all_on** — SKIPPED
  - reason: prerequisite 'only_up_all_on' success_rate=0.25 < threshold=0.8
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down

Finished: 2026-09-18 15:26:27 UTC
