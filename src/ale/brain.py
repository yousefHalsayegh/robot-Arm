"""
the RL part
"""
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from random import random, randint, sample
from collections import namedtuple, deque

import config
plt.ion()
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Brain():
    def __init__(self):
        #the network aspect
        self.policy = Network().to("cuda")
        self.optimiser = optim.Adam(self.policy.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.test = Network().to("cuda")
        self.test.eval()

        self.buffer = ReplayBuffer()

        #the loss graph
        self.losses = []
        self.fig, self.ax = plt.subplots(num="Loss Curve", clear=True)
        self.ax.set_xlabel('Num Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Loss Curve')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()

    def train(self):

       if len(self.buffer) < config.WARMUP:
           return None
       
       batch = self.buffer.sample(config.BATCH)
       states = torch.FloatTensor(np.array([t.state      for t in batch])).to("cuda")
       actions = torch.LongTensor(np.array([t.action      for t in batch])).to("cuda")
       rewards = torch.FloatTensor(np.array([t.reward      for t in batch])).to("cuda")
       next_states = torch.FloatTensor(np.array([t.next_state      for t in batch])).to("cuda")
       dones = torch.FloatTensor(np.array([t.done      for t in batch])).to("cuda")
       
       q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
       with torch.no_grad():
           next_q = self.test(next_states).max(1)[0]
           targets = rewards +config.GAMMA * next_q * (1 - dones)


       loss = self.loss_fn(q_values, targets)
       self.optimiser.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
       self.optimiser.step()
       self.soft_update()
       return loss.item()
    
    def soft_update(self):
        for target_param, policy_param in zip(
            self.test.parameters(),
            self.policy.parameters()
        ):
            target_param.data.copy_(
                config.TAU * policy_param.data + 
                (1.0 - config.TAU) * target_param.data
            )

    def predict_next_action(self, state, steps, env):
        eps = config.EPS_END + (config.EPS_START - config.EPS_END) * max(0, (config.EPS_DECAY - steps) / config.EPS_DECAY)

    
        if random() < eps:
            return env.action_space.sample()

        with torch.no_grad():
            state_next = torch.FloatTensor(state).unsqueeze(0).to("cuda")
            return self.policy(state_next).argmax(dim=1).item()

    def save_checkpoint(self, episode, steps):

        torch.save(
            {
                "steps" : steps,
                "episode" : episode, 
                "policy" : self.policy.state_dict(),
                "test" : self.test.state_dict(),
                "optimizer" : self.optimiser.state_dict()
            }, f"checkpoint/brain{episode}.pth"
        )
    def save(self):
        torch.save(self.policy.state_dict(), "brain.pth")

    def load_checkpoint(self,path):
        checkpoint = torch.load(path,  map_location="cuda")

        self.policy.load_state_dict(checkpoint["policy"])
        self.test.load_state_dict(checkpoint["test"])
        self.optimiser.load_state_dict(checkpoint["optimizer"])

        return checkpoint["steps"], checkpoint["episode"]
    
class Network(nn.Module):

    #TODO add the blocks methods so I can test out which structure is the best 
    # Initialise
    def __init__(self, n_actions=6):
        super(Network, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # Forward pass
    def forward(self, input):

        return self.fc(self.conv(input))
    
class ReplayBuffer:

    
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    