# Ablation sweep summary (final merged)

Sweep 1: 2026-09-16 19:47:30 UTC -> 2026-09-17 17:08:37 UTC
Sweep 2 (resume): 2026-09-17 22:00:00 UTC -> 2026-09-18 15:26:27 UTC
Sweep 3 (resume): 2026-09-18 19:36:24 UTC -> 2026-09-19 12:50:34 UTC

## No camera or Curr

- **only_up_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5011 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=up, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/0eazl7hh
- **only_left_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5004 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=left, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/mpzz77ec
- **only_right_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5013 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=right, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/8y98lqfy
- **only_down_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5003 >= target 5000)
  - camera=false, curriculum=false, multi_task=false, single_cmd=down, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/rqsbqlms
- **left_up_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5026 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/btk9o9wx
- **up_down_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5037 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/qzvow0t4
- **all_no_camera_or_curr** — ALREADY COMPLETE (checkpoint at episode 5000 >= target 5000)
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/u04ipril

## No Camera

- **only_up_no_camera** — ALREADY COMPLETE (checkpoint at episode 5052 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=up, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/qugrj31h
- **only_left_no_camera** — ALREADY COMPLETE (checkpoint at episode 5012 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=left, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/a7b5btnn
- **only_right_no_camera** — ALREADY COMPLETE (checkpoint at episode 5005 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=right, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/ozwg9j93
- **only_down_no_camera** — ALREADY COMPLETE (checkpoint at episode 5047 >= target 5000)
  - camera=false, curriculum=true, multi_task=false, single_cmd=down, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/kd9x54rj
- **left_up_no_camera** — ALREADY COMPLETE (checkpoint at episode 5255 >= target 5000)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/3xylo330?nw=nwuseryousefhalsayegh
- **up_down_no_camera** — ALREADY COMPLETE (checkpoint at episode 5151 >= target 5000)
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/a2yhtgxx?nw=nwuseryousefhalsayegh
- **all_no_camera** — completed
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=-, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/vgil7tlr

## No Curr

- **only_up_no_curr** — completed
  - camera=true, curriculum=false, multi_task=false, single_cmd=up, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/9tncxedb
- **only_left_no_curr** — completed
  - camera=true, curriculum=false, multi_task=false, single_cmd=left, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/am3pp9tr
- **left_up_no_curr** — ALREADY COMPLETE (checkpoint at episode 5001 >= target 5000)
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/6176n61k
- **up_down_no_curr** — ALREADY COMPLETE (checkpoint at episode 5000 >= target 5000)
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/9utw2393
- **all_no_curr** — ALREADY COMPLETE (checkpoint at episode 5000 >= target 5000)
  - camera=true, curriculum=false, multi_task=true, single_cmd=-, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/aj3smfe6

## All on

- **only_up_all_on** — ALREADY COMPLETE (checkpoint at episode 5000 >= target 5000)
  - camera=true, curriculum=true, multi_task=false, single_cmd=up, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/9bfc5aoo
- **only_left_all_on** — ALREADY COMPLETE (checkpoint at episode 5000 >= target 5000)
  - camera=true, curriculum=true, multi_task=false, single_cmd=left, task_subset=-
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/97hydrx1
- **all_all_on** — completed
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=-, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/5xf4cskv

## Task subset

- **left_up_all_on** — ALREADY COMPLETE (checkpoint at episode 5010 >= target 5000)
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/7n6ykkk6
- **up_down_all_on** — completed
  - camera=true, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down, num_envs=64
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/5mfnusut