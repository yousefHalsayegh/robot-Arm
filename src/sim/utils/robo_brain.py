import numpy as np 
import torch
import os
import torch.nn as nn
from collections import deque, namedtuple
from random import random
import torch.optim as optim
import torchvision.models as models

import ale.config as config

Transition = namedtuple('Transition', ['camera_state', 'joint_state', 'action', 'reward', 'camera_next', 'joint_next', 'done', 'n'])

class Brain:
    
    def __init__(self,ce=256, je=64,act=5, lr=0, wp=0, b=0, g=0, tau=0, ee=0, es=0, ed=0,c=0):
        
        #initializing the policy network
        self.policy = ConnectingNetwork(ce,je, act).to("cuda")

        #initializing the test network
        self.target = ConnectingNetwork(ce,je, act).to("cuda")

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

        #parameters and the replay buffer
        self.buffer = ReplayBuffer(c)

        self.eps = es
        self.warmup = wp
        self.batch = b
        self.gamma = g
        self.tau = tau
        self.eps_end = ee
        self.eps_start = es
        self.eps_decay = ed

        self.norm_success = RunningNormaliser()
        self.norm_penalty = RunningNormaliser()
        self.norm_bonus   = RunningNormaliser()

    def normalise_rewards(self, success, penalty, bonus):
        self.norm_success.update(success)
        self.norm_penalty.update(penalty)
        self.norm_bonus.update(bonus)

        return (
            self.norm_success.normalise(success)
            + self.norm_penalty.normalise(penalty)
            + self.norm_bonus.normalise(bonus)
        )

    def predict_next_action(self, camera_state, joint_state, steps, env):
        """
        Calculate the next action, given the current state, and epsilon
        """
        self.eps = self.eps_end+ (self.eps_start - self.eps_end) * max(0, (self.eps_decay - steps) / self.eps_decay)

        #epsilon is used to add some randomnes to the enviroment, if epsilon is big it is more likely that the env choose a random action, otherwise it is computed by the network
        if random() < self.eps:
            sample = env.action_space.sample()
            return int(sample[0]) if hasattr(sample, '__len__') else int(sample)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to("cuda")
            return self.policy(state_t).argmax(dim=1).item()

class ConnectingNetwork(nn.Module):

    def __init__(
        self,
        cam_embedding_size:   int = 256,
        joint_embedding_size: int = 64,
        n_actions:            int = 5,
    ):
        super().__init__()
        self.camera_net = CameraNetwork(cam_embedding_size)
        self.joint_net  = JointsNetwork(joint_embedding_size)
 
        fused_size = cam_embedding_size + joint_embedding_size
        self.policy_head = nn.Sequential(
            nn.Linear(fused_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
 
    def forward(
        self,
        camera_obs: torch.Tensor,   # [batch, 16, 224, 224]
        joint_obs:  torch.Tensor,   # [batch, 6]
    ) -> torch.Tensor:

        cam_emb   = self.camera_net(camera_obs)
        joint_emb = self.joint_net(joint_obs)
        fused     = torch.cat([cam_emb, joint_emb], dim=1)
        return self.policy_head(fused)


class JointsNetwork(nn.Module):

    def __init__(self, joint_embedding_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, joint_embedding_size),
            nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)


class CameraNetwork(nn.Module):


    def __init__(self, cam_embedding_size: int = 256):
        super().__init__()
 
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
 
        # replace first conv: 3-channel → 16-channel, same kernel/stride/padding
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels  = 16,
            out_channels = original_conv.out_channels,
            kernel_size  = original_conv.kernel_size,
            stride       = original_conv.stride,
            padding      = original_conv.padding,
            bias         = False,
        )
        # initialise new first layer — do not copy pretrained weights
        # since channel count changed
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out",
                                nonlinearity="relu")
 
        # remove the final classification head
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cam_embedding_size),
            nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.backbone(x)   # [batch, 512, 1, 1]
        return self.embed_head(features)


class RunningNormaliser:
    

    def __init__(self, epsilon = 1e-8):
        
        self.mean = 0
        self.var = 1.0
        self.count = 0

        self.epsilon = epsilon


    def update(self, value):
        self.count += 1

        delta = value - self.mean
        self.mean += delta/self.count
        self.var = self.var + delta * (value - self.mean)

    def normalise(self, value):
        std = np.sqrt(self.var/ max(self.count,1)) + self.epsilon
        return value/std
    

    def state_dict(self):
        return{
            "mean" : self.mean,
            "var" : self.var,
            "count" : self.count 
        }
    
    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]

class ReplayBuffer:

    """
    Used to save observations for the CNN and then sampled from for training purposes 
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity  = int(capacity)
        self.alpha     = alpha
        self.beta      = beta
        self.buffer    = []
        self.priorities = np.zeros(int(capacity), dtype=np.float32)
        self.pos       = 0

    def push(self, *args):
        # new transitions get max priority so they are sampled at least once
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n           = len(self.buffer)
        priorities  = self.priorities[:n]
        probs       = priorities ** self.alpha
        probs      /= probs.sum()
        
        indices     = np.random.choice(n, batch_size, replace=False, p=probs)
        samples     = [self.buffer[i] for i in indices]
        
        # importance sampling weights — correct for sampling bias
        weights     = (n * probs[indices]) ** (-self.beta)
        weights    /= weights.max()
        
        return samples, indices, torch.FloatTensor(weights).to("cuda")

    def update_priorities(self, indices, td_errors):
        """Call after train() with the computed TD errors."""
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + 1e-6   # small epsilon avoids zero priority

    def __len__(self):
        return len(self.buffer)