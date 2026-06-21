"""
the RL part
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
from collections import namedtuple, deque
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Brain():
    def __init__(self, lr=0, wp=0, b=0, g=0, tau=0, ee=0, es=0, ed=0,c=0):
        #fix it to work in both situations
        self.policy = Network().to("cuda")
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.test = Network().to("cuda")
        self.test.eval()

        self.buffer = ReplayBuffer(c)
        self.eps = 0
        self.warmup = wp
        self.batch = b
        self.gamma = g
        self.tau = tau
        self.eps_end = ee
        self.eps_start = es
        self.eps_decay = ed
    
    def train(self):

       if len(self.buffer) < self.warmup:
           return 0, 0
       
       batch = self.buffer.sample(self.batch)
       states = torch.FloatTensor(np.array([t.state      for t in batch])).to("cuda")
       actions = torch.LongTensor(np.array([t.action      for t in batch])).to("cuda")
       rewards = torch.FloatTensor(np.array([t.reward      for t in batch])).to("cuda")
       next_states = torch.FloatTensor(np.array([t.next_state      for t in batch])).to("cuda")
       dones = torch.FloatTensor(np.array([t.done      for t in batch])).to("cuda")
       
       q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
       with torch.no_grad():
           next_q = self.test(next_states).max(1)[0]
           targets = rewards +self.gamma* next_q * (1 - dones)


       loss = self.loss_fn(q_values, targets)
       self.optimiser.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
       self.optimiser.step()
       self.soft_update()

       grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in self.policy.parameters()
        if p.grad is not None
        ) ** 0.5


       return loss.item(), grad_norm
    
    def soft_update(self):
        for target_param, policy_param in zip(
            self.test.parameters(),
            self.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + 
                (1.0 - self.tau) * target_param.data
            )

    def predict_next_action(self, state, steps, env):
        self.eps = self.eps_end+ (self.eps_start - self.eps_end) * max(0, (self.eps_decay - steps) / self.eps_decay)

        if random() < self.eps:
            return env.action_space.sample()

        with torch.no_grad():
            state_next = torch.FloatTensor(state).to("cuda")
            return self.policy(state_next).argmax(dim=1).cpu().numpy().astype(np.int64)

    def save_checkpoint(self, episode, steps, path="checkpoint"):

        torch.save(
            {
                "steps" : steps,
                "episode" : episode, 
                "policy" : self.policy.state_dict(),
                "test" : self.test.state_dict(),
                "optimizer" : self.optimiser.state_dict()
            }, f"{path}/Checkpoints/brain{episode}.pth"
        )
    def save(self, path):
        torch.save(self.policy.state_dict(), f"{path}/brain.pth")

    def load_checkpoint(self,path):
        checkpoint = torch.load(path,  map_location="cuda")

        self.policy.load_state_dict(checkpoint["policy"])
        self.test.load_state_dict(checkpoint["test"])
        self.optimiser.load_state_dict(checkpoint["optimizer"])

        return checkpoint["steps"], checkpoint["episode"]
    
    def rollout(self, state):
        with torch.no_grad():
            state_next = torch.FloatTensor(state).unsqueeze(0).to("cuda")
            return self.policy(state_next).argmax(dim=1).item()
        
    def ball_position(self,obs):

        court =obs[14:76, 16:79]
        player = obs[14:76, 70:79]
        ball_pixels = np.argwhere((court > 0.7) & (court < 0.9))

        ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None

        paddle_pixels = np.argwhere((player > 0.4) & (player < 0.9))

        paddle_y = float(np.mean(paddle_pixels[:, 0])) if len(paddle_pixels) > 0 else None
        return ball_y, paddle_y

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

    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    