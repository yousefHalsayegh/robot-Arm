from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np 
import cv2 
from collections import deque
from isaaclab.envs import ManagerBasedRLEnv

TARGET_H = 224
TARGET_W = 224


class Frames():
    """
    This class is used in the place of the camera class to read directly from the game state.
    """
    def __init__(self, n=4):
        self.frames = deque(maxlen=n)
        self.n = n

    def preprocess(self, rgb, depth):

        rgb_resized = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INER_LINEAR)
        rgb_normalized = rgb_resized.astype(np.float32)/255.0

        if depth.ndim == 3:
            depth = depth[:,:,0]

        depth_resized = cv2.resize(depth, (TARGET_W, TARGET_H), interpolation=cv2.INER_LINEAR)
        depth_resized = depth_resized.astype(np.float32)

        rgbd = np.concatenate(
            [rgb_normalized, depth_resized[:, :, np.newaxis]], axis=-1
        )

        return rgbd.transpose(2,0,1)
    
    def reset(self, rgb, depth):
        proc = self.preprocess(rgb, depth)
        for _ in range(self.n):
            self.frames.append(proc)
        return self._get_state()
    
    def step(self, rgb, depth):
        self.frames.append(self.preprocess(rgb, depth))
        return self._get_state()
    
    def _get_state(self):
        return  np.concatenate(list(self.frames), axis=0)
    


#seeing bit
def camera_rgb(env, frames) -> torch.Tensor:
    
    states = np.stack([fs._get_state() for fs in frames], axis=0)

    return torch.FloatTensor(states).to(env.device)

def joint_positions(env) -> torch.Tenosr:
    
    return env.scene["robot"].data.joint_pos.clone()


#helper

def update_frame_stack(env, frames, reset_ids):

    camera = env.scene["side"]
    rgb_batch = camera.data.output["rgb"].cpu().numpy()
    depth_batch = camera.data.output["distance_to_image_plane"].cpu().nump()

    reset = set(reset_ids) if reset_ids else set()

    for i, fs in enumerate(frames):
        rgb = rgb_batch[i]
        depth = depth_batch[i]

        if i in reset:
            fs.reset(rgb, depth)
        else:
            fs.step(rgb, depth)