# Ablation sweep summary

Started: 2026-09-15 15:58:25 UTC


## No camera or Curr

- **only_down_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=false, single_cmd=down, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/5291znv9
  - log: `ablation_logs/only_down_no_camera_or_curr.log`
- **only_up_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=false, single_cmd=up, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/vp8arzjm
  - log: `ablation_logs/only_up_no_camera_or_curr.log`
- **only_left_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=false, single_cmd=left, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/86wjri53
  - log: `ablation_logs/only_left_no_camera_or_curr.log`
- **only_right_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=false, single_cmd=right, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/v3ueh22e
  - log: `ablation_logs/only_right_no_camera_or_curr.log`
- **all_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/o62ize2u
  - log: `ablation_logs/all_no_camera_or_curr.log`
  - **left_up_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=left,up, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/4b9l0z1n
  - log: `ablation_logs/left_up_no_camera_or_curr.log`
- **up_down_no_camera_or_curr** — completed
  - camera=false, curriculum=false, multi_task=true, single_cmd=-, task_subset=up,down, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/b5neb9bz
  - log: `ablation_logs/up_down_no_camera_or_curr.log`

## No Camera

- **only_left_no_camera** — completed
  - camera=false, curriculum=true, multi_task=false, single_cmd=left, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/oi3ir8st
  - log: `ablation_logs/only_left_no_camera.log`
- **only_up_no_camera** — completed
  - camera=false, curriculum=true, multi_task=false, single_cmd=up, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/ecgboob7
  - log: `ablation_logs/only_up_no_camera.log`
- **all_no_camera** — completed
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/vgil7tlr
  - log: `ablation_logs/all_no_camera.log`

  - **left_up_no_camera** — completed
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=left,up, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/zo9z9qqe
  - log: `ablation_logs/left_up_no_camera.log`
- **up_down_no_camera** — completed
  - camera=false, curriculum=true, multi_task=true, single_cmd=-, task_subset=up,down, num_envs=1024
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/lpgf39tb
  - log: `ablation_logs/up_down_no_camera.log`

## No Curr

- **only_up_no_curr** — completed
  - camera=true, curriculum=false, multi_task=false, single_cmd=up, task_subset=-, num_envs=16
  - run: https://wandb.ai/models-imperial-college-london6785/RL%20for%20Games/runs/d5voddr1
  - log: `ablation_logs/only_up_no_curr.log`
- **only_left_no_curr** — completed
  - camera=true, curriculum=false, multi_task=false, single_cmd=left, task_subset=-, num_envs=16
  - run: (no wandb URL found in log — check ablation_logs/only_left_no_curr.log)
  - log: `ablation_logs/only_left_no_curr.log`


