"""
lerobot_recorder.py

Records Isaac Sim training episodes in LeRobot-compatible format.
Each timestep stores joint positions, camera frame, and action target.

Output structure:
    lerobot_dataset/
        data.jsonl          — scalar data per frame
        images/side/        — PNG frames from the side camera
        meta.json           — dataset metadata
"""

import json
import numpy as np
import cv2
from pathlib import Path


class LeRobotRecorder:
    """
    Records episodes during sim training for later use with
    LeRobot RL or VLA fine-tuning on the physical SO101.
    """

    def __init__(self, output_dir: str, task_name: str = "pong_arm"):
        self.output_dir    = Path(output_dir)
        self.task_name     = task_name
        self.episode_index = 0
        self.global_index  = 0
        self.ep_buffer     = []
        self.buffer        = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "side").mkdir(parents=True, exist_ok=True)

    def record_step(
        self,
        joint_pos:    np.ndarray,
        joint_target: np.ndarray,
        side_frame:   np.ndarray,
        frame_index:  int,
        timestamp:    float,
        done:         bool,
    ):
        self.buffer.append({
            "observation.state":       joint_pos.tolist(),
            "observation.images.side": side_frame,
            "action":                  joint_target.tolist(),
            "timestamp":               timestamp,
            "frame_index":             frame_index,
            "episode_index":           self.episode_index,
            "index":                   self.global_index,
            "next.done":               done,
            "task_index":              0,
        })
        self.global_index += 1

    def end_episode(self):
        self.ep_buffer.extend(self.buffer)
        self.buffer        = []
        self.episode_index += 1

    def save(self):
        img_dir = self.output_dir / "images" / "side"
        records = []

        for frame in self.ep_buffer:
            ep   = frame["episode_index"]
            fidx = frame["frame_index"]
            img_path = img_dir / f"episode_{ep:06d}_frame_{fidx:06d}.png"

            cv2.imwrite(
                str(img_path),
                cv2.cvtColor(frame["observation.images.side"], cv2.COLOR_RGB2BGR)
            )

            records.append({
                "observation.state":       frame["observation.state"],
                "observation.images.side": str(img_path),
                "action":                  frame["action"],
                "timestamp":               frame["timestamp"],
                "frame_index":             frame["frame_index"],
                "episode_index":           frame["episode_index"],
                "index":                   frame["index"],
                "next.done":               frame["next.done"],
                "task_index":              frame["task_index"],
            })

        with open(self.output_dir / "data.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        meta = {
            "task":           self.task_name,
            "total_episodes": self.episode_index,
            "total_frames":   self.global_index,
            "fps":            30,
            "robot_type":     "so101",
            "cameras":        ["side"],
            "joints": [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex",   "wrist_roll",    "gripper",
            ],
        }
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"saved {self.episode_index} episodes, "
              f"{self.global_index} frames → {self.output_dir}")