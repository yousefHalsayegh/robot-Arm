# Vision-Language-Action Control of a Robotic Arm

**MSc Project | Imperial College London**  
**Python | NVIDIA Isaac Lab | LeRobot | SmolVLA | Soft Actor-Critic | Rainbow DQN | Imitation Learning**

A robotics research project investigating how a robotic arm can interact with visual interfaces through a physical joystick.

The system combines reinforcement learning, imitation learning, simulation, vision-based observations, and physical robot control. Video games are used as evaluation environments because they provide clear task objectives while allowing the robotic controller to be tested against changing visual interfaces.

The broader objective is to move toward a system that can be shown a different interface and still act through a small set of learned commands, rather than relying on task-specific hard-coded control logic for every screen.

## Project Overview

This project was developed as my **MSc final project at Imperial College London**.

The system is built around the SO-ARM platform and NVIDIA Isaac Lab. The work includes:

- a simulated robotic-arm environment;
- a digital twin of the physical control setup;
- a custom joystick asset;
- reinforcement-learning agents for both game playing and low-level robot control;
- LeRobot-compatible demonstration-data collection;
- imitation-learning workflows;
- physical SO-ARM experiments;
- sim-to-real development.

The repository is **based on** the upstream Isaac Lab SO-ARM100 / SO-ARM101 project by Le Lay and Bay, but the project code was substantially reworked for this research direction.

## Research Goal

The central question is whether a robotic system can learn to operate a physical interface in a way that transfers across visual tasks.

The experimental setup separates the problem into two levels:

1. **Game / task policy** — determines the desired action for the current game state.
2. **Robot control policy** — converts a high-level command into physical joint-level behaviour that manipulates the joystick.

This decomposition allows the robot controller to focus on simple commands such as moving the joystick in the required direction, rather than requiring the robot itself to understand every game-specific screen.

The longer-term goal is for the same physical controller to work when presented with a previously unseen interface, provided that another policy or decision layer can map that interface to the appropriate command.

## Learning Approaches

Several learning approaches were investigated during the project.

### Rainbow DQN

A **Rainbow DQN** agent was implemented based on the Rainbow DQN paper for the Atari Learning Environment side of the project.

The implementation includes core deep-RL infrastructure such as:

- replay buffer;
- target network;
- soft target updates;
- epsilon-based exploration;
- discounted rewards;
- n-step returns;
- checkpointing;
- configurable replay capacity and warm-up;
- Weights & Biases logging.

The current ALE implementation focuses on **Pong**.

The game-playing RL policy was successfully trained during the project.

### Soft Actor-Critic

**Soft Actor-Critic (SAC)** was developed for continuous, low-level robotic-arm control.

The controller operates at the joint level and uses observations including:

- joint states;
- RGB images;
- depth information.

The robot-control reward combines several objectives, including:

- distance to the joystick;
- per-step penalty;
- correct joystick movement;
- correct gripper behaviour;
- correct wrist-joint behaviour.

The SAC robotic-control experiments were still under development at the end of the project and did not reach the same level of successful policy training as the ALE Rainbow DQN experiments.

### Imitation Learning

Imitation learning is used as part of the robot-control workflow.

A LeRobot-compatible data pipeline supports demonstration data and replay-buffer pre-filling, allowing recorded or generated trajectories to be used to bootstrap learning.

### SmolVLA

**SmolVLA** was investigated as an initial vision-language-action approach.

It was used as a starting point for the VLA direction of the project, but it was later dropped because its inference/control speed was not suitable for the required interaction loop.

The project therefore shifted toward more specialised reinforcement-learning and control approaches for the main experiments.

## Simulation and Digital Twin

The robotic system was modelled in **NVIDIA Isaac Lab / Isaac Sim**.

The simulation includes a digital representation of the physical manipulation setup and a custom joystick asset created for the project.

This environment provides a controlled space for:

- robot-control training;
- action testing;
- reward development;
- data collection;
- synthetic trajectory generation;
- sim-to-real experimentation.

The project was also run on the physical SO-ARM system, so sim-to-real was an explicit project objective rather than only a future possibility.

## Environments

The project defines custom Isaac Lab environments including:

- `Fighter`
- `Player`

These environments support different parts of the robotic-control and game-playing workflow.

Use:

```bash
list_envs
```

to inspect the environments exposed by the package.

## Baseline Agents

### Zero Agent

```bash
zero_agent
```

The current zero-agent script is primarily used to test the stiffness/behaviour of the joystick object in simulation.

### Random Agent

```bash
random_agent
```

Runs a simple random-action baseline.

These scripts are useful for validating the environment independently of a trained policy.

## ALE / Pong Agent

The `src/ale` component contains the reinforcement-learning implementation focused on playing **Pong**.

The training/playback interface exposes configuration for:

- task selection;
- number of environments;
- total episodes;
- updates per episode;
- model-save frequency;
- learning rate;
- replay-buffer warm-up;
- batch size;
- soft-update coefficient (`tau`);
- epsilon start/end/decay;
- discount factor (`gamma`);
- replay-buffer capacity;
- checkpoints;
- LeRobot recording;
- frame rate;
- Weights & Biases logging.

Example entry point:

```bash
play --task <TASK>
```

Additional options can be inspected with:

```bash
play --help
```

## Low-Level Joystick SAC Training

The robot-control training command is designed for low-level joystick manipulation.

```bash
train --task <TASK>
```

Important options include:

```text
--num_envs
--episodes
--full_save
--mid_save
--checkpoint
--capacity
--wandb
--cam_embedding
--joint_embedding
--decision_steps
--lerobot_repo_id
--synthetic_per_cmd
--action_scale_deg
--prefill_path
--prefill
```

The default configuration uses multiple parallel simulation environments and supports replay-buffer pre-filling before live SAC training begins.

Use:

```bash
train --help
```

for the complete argument list available in the current code.

## Replay Buffer and Demonstration Data

The project includes a dedicated buffer-generation workflow:

```bash
buffer --task <TASK>
```

The buffer is used for:

- RL replay;
- imitation learning;
- pre-filling the SAC replay buffer.

It can work with an existing LeRobot dataset and can generate synthetic trajectories.

Important options include:

```text
--lerobot_repo_id
--output_path
--synthetic_per_cmd
--interp_steps
--decision_steps
--action_scale_deg
```

By default, the generated replay data is written to:

```text
buffer_prefill.pkl
```

The `decision_steps` and `action_scale_deg` values used for generated trajectories are intended to match the corresponding live controller settings.

## Project Structure

```text
robot-Arm/
├── src/
│   ├── ale/
│   ├── assets/
│   └── sim/
├── pyproject.toml
├── uv.lock
├── CITATION.cff
└── README.md
```

### `src/ale`

Contains the reinforcement-learning code focused on Pong and the ALE-side Rainbow DQN experiments.

### `src/assets`

Contains simulation assets used by the project, including the custom joystick developed for the experimental setup.

### `src/sim`

Contains the Isaac Lab simulation environments, robot-control code, training workflows, data-buffer tools, and related project implementation.

## Installation

The package configuration currently specifies:

- Python 3.11
- NVIDIA Isaac Lab 2.3.0
- Isaac Sim
- PyTorch 2.7.0
- Torchvision 0.22.0
- `uv`

Clone the repository:

```bash
git clone https://github.com/yousefHalsayegh/robot-Arm.git
cd robot-Arm
```

Then install the Python environment:

```bash
uv sync
```

> **Important:** these versions come from the repository configuration. The setup instructions have not yet been validated from a completely clean machine.

Isaac Sim and Isaac Lab also require NVIDIA-specific system dependencies, including a compatible GPU and driver environment. The repository does not currently document the exact host OS, GPU, CUDA, and driver combination used during development.

## Results

The project reached different levels of maturity across its components.

### Successfully demonstrated

- Rainbow DQN training for the Pong/ALE component
- Isaac Lab simulation environments
- custom joystick simulation asset
- replay-buffer and demonstration-data workflows
- physical SO-ARM operation
- sim-to-real experimentation infrastructure

### Still experimental

- fully trained SAC policy for reliable robotic joystick manipulation
- broader generalisation across unseen interfaces
- end-to-end VLA control

Weights & Biases experiment plots were produced during development but are intentionally not included in this public README.

## My Contribution

This repository is based on the upstream **Isaac Lab – SO-ARM100 / SO-ARM101 Project**, but the project was substantially reworked for my MSc research.

I changed or implemented the project files used for my experimental system, with the main exceptions being upstream material such as:

- citation metadata;
- `enhance/`;
- `robots/`.

My work includes the reinforcement-learning systems, the ALE/Pong work, SAC controller development, joystick asset, simulation environments and adaptations, data-buffer workflows, imitation-learning integration, physical-control experiments, and the overall research pipeline used for the MSc project.

## Upstream Attribution

The repository's citation metadata references:

**Isaac Lab – SO-ARM100 / SO-ARM101 Project**  
Le Lay, Louis and Bay, Muammer  
Upstream repository: `MuammerBay/isaac_so_arm101`  
License listed in the upstream citation metadata: BSD-3-Clause

This repository should be described as **based on** that project rather than as an entirely independent implementation.

## Current Limitations

- The SAC robot-control policy was not fully successful at the end of the project.
- Clean-machine installation has not yet been validated.
- Exact host GPU / CUDA / driver setup is not currently documented.
- SmolVLA was investigated but dropped because it was too slow for the intended control loop.
- Generalisation to unseen interfaces remains a research objective rather than a fully demonstrated result.
- Public W&B plots and experiment dashboards are intentionally omitted.

## Intended Audience

This repository is intended for:

- robotics and reinforcement-learning researchers;
- students exploring Isaac Lab, LeRobot, and robot learning;
- recruiters or engineers reviewing the technical scope of the MSc project.

## License

No license is currently specified for this repository.
