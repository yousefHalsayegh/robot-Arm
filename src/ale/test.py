"""
Finetune only 'up' and 'down' in positions.json using keyboard jogging --
no leader arm needed. 'home' and 'neutral' are taken from the existing
saved file and used only to move the arm through a home -> neutral
startup sequence; they are not modified by this script.

Controls (single keypress, no need to hit Enter):
  [ / h / left arrow   -- select previous joint
  ] / l / right arrow  -- select next joint
  up arrow / k         -- increase selected joint's value by current step
  down arrow / j       -- decrease selected joint's value by current step
  +                    -- double the step size
  -                    -- halve the step size
  s                    -- save the current preset to positions.json now
  n                    -- move on to the next preset (auto-saves first)
  q                    -- quit (auto-saves current preset first)
"""
import json
import os
import sys
import termios
import tty

from robot_noVla import Robot

POSITIONS_FILE = "positions.json"
PRESET_ORDER = ["up", "down"]  # only these are edited/saved by this script
DEFAULT_STEP = 1.0
MIN_STEP = 0.01

ARROW_UP = '\x1b[A'
ARROW_DOWN = '\x1b[B'
ARROW_LEFT = '\x1b[D'
ARROW_RIGHT = '\x1b[C'


def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {"home": {}, "neutral": {}, "up": {}, "down": {}}


def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def read_key():
    """
    Reads a single keypress from stdin without waiting for Enter. Arrow
    keys arrive as a 3-byte escape sequence, read as a group; everything
    else is a single character.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def edit_preset(robot, preset_name, joint_values):
    """
    Interactive jog loop for one preset. Mutates joint_values in place.
    Returns 'next', 'quit', or 'save_only' depending on how the user exits.
    """
    joint_names = list(joint_values.keys())
    if not joint_names:
        # nothing saved for this preset yet -- seed from the robot's
        # current live position so there's something to edit
        obs = robot.rb.get_observation()
        joint_values.update({k: v for k, v in obs.items() if '.pos' in k})
        joint_names = list(joint_values.keys())

    idx = 0
    step = DEFAULT_STEP

    def send_current():
        try:
            robot.rb.send_action(joint_values)
        except Exception as e:
            print(f"\n(couldn't send action: {e})")

    print(f"\n--- editing '{preset_name}' ---")
    send_current()

    while True:
        joint = joint_names[idx]
        print(f"\r[{preset_name}] {joint} = {joint_values[joint]:.3f}  (step={step:g})    ", end="", flush=True)
        key = read_key()

        if key in ('[', 'h', ARROW_LEFT):
            idx = (idx - 1) % len(joint_names)
        elif key in (']', 'l', ARROW_RIGHT):
            idx = (idx + 1) % len(joint_names)
        elif key in (ARROW_UP, 'k'):
            joint_values[joint] += step
            send_current()
        elif key in (ARROW_DOWN, 'j'):
            joint_values[joint] -= step
            send_current()
        elif key == '+':
            step *= 2
        elif key == '-':
            step = max(step / 2, MIN_STEP)
        elif key == 's':
            print(f"\nSaved '{preset_name}'.")
            return "save_only"
        elif key == 'n':
            return "next"
        elif key == 'q':
            return "quit"


def main():
    robot = Robot()  # connects the follower arm only -- no leader/teleop
    positions = load_positions()
    robot.positions = positions

    print("Moving from 'home' to 'neutral' using the saved positions...")
    robot.inital()  # sends the saved 'home' position, waits until settled
    robot.reset()   # sends the saved 'neutral' position, waits until settled
    print("Ready.\n")

    print("Editing 'up' and 'down' only -- 'home'/'neutral' are untouched.")
    print("Use arrow keys / h,j,k,l to select/nudge joints, +/- for step,")
    print("s to save, n for next preset, q to quit.\n")

    try:
        i = 0
        while i < len(PRESET_ORDER):
            preset_name = PRESET_ORDER[i]
            joint_values = positions.setdefault(preset_name, {})

            result = edit_preset(robot, preset_name, joint_values)

            if result == "save_only":
                save_positions(positions)
                continue  # stay on the same preset
            elif result == "next":
                save_positions(positions)
                i += 1
            elif result == "quit":
                save_positions(positions)
                break
    except KeyboardInterrupt:
        pass
    finally:
        save_positions(positions)
        print("\nFinal positions saved to positions.json.")


if __name__ == "__main__":
    main()