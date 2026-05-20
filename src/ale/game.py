"""
the ALE part
"""

import inputs
import numpy as np
import gymnasium as gym
import ale_py



gym.register_envs(ale_py)

CONTROLLER_TO_ACTION = {
    ("ABS_HAT0X", -1): "LEFT",
    ("ABS_HAT0X",  1): "RIGHT",
    ("ABS_HAT0Y", -1): "UP",
    ("ABS_HAT0Y",  1): "DOWN",
    ("BTN_SOUTH",  1): "FIRE",       # A 
    ("BTN_EAST",   1): "FIRE",       # B 
    ("BTN_NORTH",  1): "FIRE",     # Y
    ("BTN_WEST",   1): "FIRE",   # X
    ("BTN_TR",     1): "FIRE",  # R1
    ("BTN_TL",     1): "FIRE",   # L1
    ("BTN_TR2",    1): "FIRE",   # R2
    ("BTN_TL2",    1): "FIRE",     # L2
}

def read_controller_action() -> tuple[str, int] | tuple[None, None]:
    events = inputs.get_gamepad()
    for e in events:
        if e.ev_type in ("Sync", "Misc"):
            continue
        action_name = CONTROLLER_TO_ACTION.get((e.code, e.state))
        if action_name:
            action_int = getattr(ale_py.Action, action_name).value
            return action_name, action_int
    return None, None

def main():
    env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=8)
    env.reset(seed=42)

    try:
        while True:
            action_name, action_int = read_controller_action()

            if action_name is None:
                continue

            # Step the ALE game
            obs, reward, terminated, truncated, info = env.step(action_int)
            # add this temporarily before the main loop in game.py


            if terminated or truncated:
                obs, info = env.reset()

    finally:
        env.close()



if __name__ == "__main__":
    main()