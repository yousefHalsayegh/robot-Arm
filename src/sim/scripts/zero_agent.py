

import argparse
from isaaclab.app import AppLauncher
import ale.config as config

parser = argparse.ArgumentParser("Isaac Sim DQN — Robot Arm Pong")
parser.add_argument("--task",type=str,default=None)
parser.add_argument("--num_envs",type=int,default=1)
parser.add_argument("-ep", "--episode", help="The amount of episodes to train for in total", type=int, default=config.EPISODES)
parser.add_argument("-f", "--frames", default=60, type=int, help="the framerate of the game")
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is up."""

import gymnasium as gym
import numpy as np
import sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from sim.utils.robot_sim import (
    RobotSim, POSITIONS,
    send_targets, batch_move_arms,
    batch_move_arm,calibrate_zone_position, measure_joystick_response
)
from sim.utils.pong_display import PongDisplay

import ale_py
gym.register_envs(ale_py)



def env_init(seed, rank):
    """Factory function for SyncVectorEnv."""
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array", repeat_action_probability=0)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True,
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed + rank)
        return env
    return _init

def joystick_zone(object_art, env_index):
    """Classify the joystick's CURRENT physical position into a zone,
    independent of what task it's being driven toward. This is what the
    game should actually see, moment to moment — same as a real controller."""
    tilt_deg = np.rad2deg(object_art.data.joint_pos[env_index].cpu().numpy())  # [PivotY, PivotX]
    axis_deg = tilt_deg[PIVOT_X_IDX]

    if axis_deg < -DEADZONE_DEG:
        return "up"
    elif axis_deg > DEADZONE_DEG:
        return "down"
    return "neutral"


ZONE_TO_ACT = {"up": 2, "down": 3, "neutral": 0}

DEADZONE_DEG = 6.0 

# joint order from your earlier print: ['PivotY', 'PivotX']
PIVOT_Y_IDX = 0
PIVOT_X_IDX = 1
NOOP, FIRE, UP, DOWN = 0, 1, 2, 3

def joystick_registered(object_art, env_index, task):
    tilt = object_art.data.joint_pos[env_index].cpu().numpy()  # [PivotY, PivotX], radians
    tilt_deg = np.rad2deg(tilt)
    if task == "neutral":
        return np.abs(tilt_deg).max() < DEADZONE_DEG

    axis_deg = tilt_deg[PIVOT_X_IDX]
    if task == "up":
        return axis_deg < -DEADZONE_DEG
    elif task == "down":
        return axis_deg > DEADZONE_DEG
    return False

def ball_position(obs):
        """
        Used for Atari Pong, using the observation of the env calculates the ball and paddle position. 
        """

        #divides the screen into where the court (mid side) and player (right side) of the screens
        court =obs[15:75, 5:80]
        player = obs[15:79, 75:84]

        #Locates the location of the ball using thresholds for the intensity then extracting the Y axis
        ball_pixels = np.argwhere((court > 0.5) & (court < 0.9))
        ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None



        #Locates the location of the player paddle using thresholds for the intensity then extracting the Y axis
        paddle_pixels = np.argwhere((player > 0.4) & (player < 0.9))
        paddle_y = float(np.mean(paddle_pixels[:, 0])) if len(paddle_pixels) > 0 else None
        return ball_y, paddle_y

def choose_action(ball_y, paddle_y, deadzone=3):
    """
    Simple proportional controller: move the paddle toward the ball's y position.
    """
    if ball_y is None or paddle_y is None:
        return FIRE  # nothing detected (e.g. ball hasn't been served yet) -> serve
 
    diff = ball_y - paddle_y
    if abs(diff) < deadzone:
        return NOOP
    elif diff < 0:
        return UP
    else:
        return DOWN


def main():

    env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=not args_cli.disable_fabric,
)
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())

    N        = args_cli.num_envs
    base_env = env.unwrapped
    device   = str(base_env.device)

    # ── ALE envs via SyncVectorEnv ────────────────────────────────
    # AtariPreprocessing + FrameStackObservation handle all preprocessing
    # obs shape out: [N, 4, 84, 84] float32 — used directly as state
    ale_envs = gym.vector.SyncVectorEnv(
        [env_init(42, i) for i in range(N)]
    )

    # obs: [N, 4, 84, 84] — states used directly, no Frames/Eyes needed
    obs, _ = ale_envs.reset()
    states  = obs.copy()   # [N, 4, 84, 84]

    # ── Pong display inside Isaac Sim ─────────────────────────────
    display = PongDisplay(num_envs=N)

    # ── articulation ──────────────────────────────────────────────
    so101  = base_env.scene["robot"]
    object_art = base_env.scene["object"]
    robots = [RobotSim(env_index=i) for i in range(N)]

    def sim_step():
        base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())
        simulation_app.update()

    # ── move all arms to home ─────────────────────────────────────
    batch_move_arms(so101, robots, sim_step, "reset", device)

    # ── per-env tracking ──────────────────────────────────────────

    current_acts  = np.zeros(N, dtype=np.int64)
    failsafe  = np.zeros(N, dtype=np.int64)
    episode = 0

    batch_move_arms(so101, robots, sim_step, "home", device)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    POSITIONS["up"]   = calibrate_zone_position(so101, object_art, 0, "up",   sim_step,
                                              PIVOT_X_IDX, DEADZONE_DEG, POSITIONS)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")


    batch_move_arms(so101, robots, sim_step, "neutral", device)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    batch_move_arms(so101, robots, sim_step, "home", device)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    POSITIONS["down"] = calibrate_zone_position(so101, object_art, 0, "down", sim_step, 
                                              PIVOT_X_IDX, DEADZONE_DEG, POSITIONS)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")

    
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    batch_move_arms(so101, robots, sim_step, "home", device)
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    physics_dt = base_env.sim.get_physics_dt()

    measure_joystick_response(so101, object_art, robots[0], sim_step, "up", PIVOT_X_IDX, DEADZONE_DEG, physics_dt)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    batch_move_arms(so101, robots, sim_step, "home", device)
    batch_move_arms(so101, robots, sim_step, "neutral", device)

    measure_joystick_response(so101, object_art, robots[0], sim_step, "down", PIVOT_X_IDX, DEADZONE_DEG, physics_dt)
    print(f"location of joystick={object_art.data.joint_pos[0].cpu().numpy() }")
    batch_move_arms(so101, robots, sim_step, "neutral", device)
    batch_move_arms(so101, robots, sim_step, "home", device)
    batch_move_arms(so101, robots, sim_step, "neutral", device)

    

    try:
     
        while episode < (args_cli.episode + 1):
            
            
            


            # ── action gating per env ─────────────────────────
            for i in range(N):
                joystick_input = joystick_registered(object_art, i, robots[i].task)
                timeout = failsafe[i] > 60

                if timeout and not joystick_input:
                    batch_move_arm(so101, robots[i], sim_step, "home", device)
                    batch_move_arm(so101, robots[i], sim_step, "neutral", device)
                    failsafe[i] = 0 

                if joystick_input :
                    
                    ball_y, paddle_y = ball_position(states[i][-1])
                    act = choose_action(ball_y, paddle_y)
                    
                    if act in (2, 4):
                        robots[i].task = "up"
                        
                    elif act in (3, 5):
                        robots[i].task = "down"
                        
                    else:
                        robots[i].task = "neutral"
                    failsafe[i] = 0 
                failsafe[i] += 1
                current_acts[i] = ZONE_TO_ACT[joystick_zone(object_art, i)]

                
                
                
                # ball_y, paddle_y = ball_position(states[i][-1])
                # current_acts[i] = choose_action(ball_y, paddle_y)
                
            sim_step()
                            
            
            # ── step ALL ALE envs at once ─────────────────────
            # returns [N, 4, 84, 84] obs — used directly as next_states
            next_obs, _, term_batch, trunc_batch, _ = \
                ale_envs.step(current_acts.copy())
            dones_batch = term_batch | trunc_batch
                

            # ── batched arm command + sim step ────────────────
            send_targets(so101, robots)

            # ── update Pong display ───────────────────────────
            for i in range(N):
                try:
                    frame = ale_envs.envs[i].render()
                    display.update(i, frame)
                except Exception:
                    pass

            # ── per-env processing ────────────────────────────
            for i in range(N):
                done_i   = bool(dones_batch[i])
                if done_i:
                    current_acts[i]   = 0
                    episode          += 1

                    # SyncVectorEnv auto-resets done envs
                    # next_obs[i] is already the fresh reset obs
                    states[i] = next_obs[i]

                    robots[i].task = "neutral"

                else:
                    states[i] = next_obs[i]


    except KeyboardInterrupt:
        print("\nclosing")
        ale_envs.close()



if __name__ == "__main__":
    main()
    