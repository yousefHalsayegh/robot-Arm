import gymnasium as gym
import ale_py
import inputs
import threading


CONTROLLER = {
    # Keep in mind this is on the X/Y trgger not DP and the values are for Pong
    ('ABS_Y', 0): 2,   # up 
    ('ABS_Y',  255): 3,   # down
    ('ABS_Y',  128): 0,   # Neutral

    # D-pad horizontal (not used in Pong but mapped anyway)
    ('ABS_X', 0): 4,   # left
    ('ABS_X',  255): 5,   # right
    ('ABS_X',  128): 0, #Neutral

    # Face buttons
    ('BTN_SOUTH', 1): 0,    # A Touch
    ('BTN_EAST',  1): 0,    # B Touch
    ('BTN_NORTH', 1): 0,    # X Touch
    ('BTN_WEST',  1): 0,    # Y Touch

    ('BTN_SOUTH', 0): 0,    # A Release
    ('BTN_EAST',  0): 0,    # B Release
    ('BTN_NORTH', 0): 0,    # X Release
    ('BTN_WEST',  0): 0,    # Y Release

    # Other "Bumpers"
    ('BTN_TR',  1): 0,      # RB Touch
    ('BTN_TL',  1): 0,      # LB Touch
    ('BTN_TR2', 1): 0,      # RT Touch
    ('BTN_TL2', 1): 0,      # LT Touch
    ('BTN_TR',  0): 0,      # RB Release
    ('BTN_TL',  0): 0,      # LB Release
    ('BTN_TR2', 0): 0,      # RT Release
    ('BTN_TL2', 0): 0,      # LT Release
}



def controller():
        global action 
     
        while True:
            try:
                events = inputs.get_gamepad()

                for e in events:
                    if e.ev_type in ("Sync", "Misc"):
                        continue
                    if e.code in ('ABS_GAS', 'ABS_BRAKE'):
                        continue

                    key = (e.code, e.state)
                    act = CONTROLLER.get(key)

                    action = act


            except Exception:
                pass

gym.register_envs(ale_py)
controller = threading.Thread(target=controller, daemon=True)
controller.start()
env = gym.make("ALE/Pong-v5", frameskip=4, render_mode="human")
env.reset(seed=42)
action = 0
while True:
    obs, raw_reward, terminated, truncated, _ = env.step(action)