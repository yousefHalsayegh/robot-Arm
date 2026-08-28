"""
the RL part
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
from collections import namedtuple, deque
import os
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'n'])

class Brain():
    """
    The class used for the RL agent
    """
    #TODO change the parameters so that it takes from the config rather than this way
    def __init__(self, lr=0, wp=0, b=0, g=0, tau=0, ee=0, es=0, ed=0,c=0, d=False, tu="soft", tup=8000, n=False):
        
        #initializing the policy network
        self.policy = Network(dueling=d, noisy=n).to("cuda")
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr, eps=1.5e-4)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        #initializing the test network
        self.test = Network(dueling=d, noisy=n).to("cuda")
        self.test.eval()

        #parameters and the replay buffer
        self.buffer = ReplayBuffer(c)
        self.eps = 0
        self.warmup = wp
        self.batch = b
        self.gamma = g
        self.tau = tau
        self.eps_end = ee
        self.eps_start = es
        self.eps_decay = ed
        self.q_value = 0
        self.target_update = tu           # "soft" or "hard"
        self.target_update_period = tup
        self.train_steps = 0

        self.agent = ""
    def train(self):
       """
       Training the policy netwrok, given the collected observations
       """

        #The warmup to allow the buffer to collect data before training starts
       if len(self.buffer) < self.warmup:
           return 0, 0
    
        #sampling different observations from the collected data
       batch, indices, weights = self.buffer.sample(self.batch)
       states = torch.FloatTensor(np.array([t.state      for t in batch])).to("cuda")
       actions = torch.LongTensor(np.array([t.action      for t in batch])).to("cuda")
       rewards = torch.FloatTensor(np.array([t.reward      for t in batch])).to("cuda")
       next_states = torch.FloatTensor(np.array([t.next_state      for t in batch])).to("cuda")
       dones = torch.FloatTensor(np.array([t.done      for t in batch])).to("cuda")
       steps = torch.FloatTensor(np.array([t.n      for t in batch])).to("cuda")
       
       #calculating the Q_Values of the collected states
       q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
       self.q_value = q_values
       with torch.no_grad():
           #calculating the approximate next Q_values and the target
           next_actions = self.policy(next_states).argmax(1, keepdim=True)
           next_q = self.test(next_states).gather(1, next_actions).squeeze(1)
           targets = rewards +(self.gamma**steps)* next_q * (1 - dones)
        
        

        #calculating the loss and passing it backward
       per_sample_loss = self.loss_fn(q_values, targets)
       loss = (weights * per_sample_loss).mean()
       td_errors = (targets - q_values).detach().cpu().numpy()
       self.optimiser.zero_grad()
       loss.backward()
       grad_norm = sum(
               p.grad.norm().item() ** 2
               for p in self.policy.parameters()
               if p.grad is not None
               ) ** 0.5
       torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
       self.optimiser.step()
       self.train_steps += 1
       if self.target_update == "soft":
            self.soft_update()
       elif self.train_steps % self.target_update_period == 0:
            self.hard_update()
       self.buffer.update_priorities(indices, td_errors)

        #calculating the grad_norm for measuring the overall performace 
       


       return loss.item(), grad_norm
    
    def soft_update(self):
        """
        Updates the test network, with a small Tau, rather than copying the full parameters
        """
        for target_param, policy_param in zip(
            self.test.parameters(),
            self.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + 
                (1.0 - self.tau) * target_param.data
            )
    def hard_update(self):
        self.test.load_state_dict(self.policy.state_dict())

    def predict_next_action(self, state, steps, env):
        """
        Calculate the next action, given the current state, and epsilon
        """
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.eps_decay - steps) / self.eps_decay)

        num_envs = state.shape[0]

        with torch.no_grad():
            state_t = torch.FloatTensor(state).to("cuda")
            greedy_action = self.policy(state_t).argmax(dim=1).cpu().numpy()

        random_mask = np.random.random(num_envs) < self.eps
        random_actions = env.action_space.sample()

        return np.where(random_mask, random_actions, greedy_action)
        
    def save_checkpoint(self, episode, steps, path="checkpoint"):
        """
        Used to save a specific spot in the training allowing continuation
        """
        torch.save(
            {
                "steps" : steps,
                "episode" : episode, 
                "policy" : self.policy.state_dict(),
                "test" : self.test.state_dict(),
                "optimizer" : self.optimiser.state_dict()
            }, f"runs/{path}/Checkpoints/brain{episode}.pth"
        )
    def save(self, path):
        """
        Save the final training step for furtherr eval 
        """
        torch.save(self.policy.state_dict(), f"runs/{path}/Checkpoints/brain.pth")

    def load_checkpoint(self,path):
        """
        Loads saved weights and steps for continuation of training
        """
        checkpoint = torch.load(path,  map_location="cuda")

        self.policy.load_state_dict(checkpoint["policy"])
        self.test.load_state_dict(checkpoint["test"])
        self.optimiser.load_state_dict(checkpoint["optimizer"])

        return checkpoint["steps"], checkpoint["episode"]
    
    def rollout(self, state):
        """
        Test the netwrok
        """
        with torch.no_grad():
            state_next = torch.FloatTensor(state).unsqueeze(0).to("cuda")
            return self.policy(state_next).argmax(dim=1).item()
        
    def ball_position(self,obs):
        """
        Used for Atari Pong, using the observation of the env calculates the ball and paddle position. 
        """

        #divides the screen into where the court (mid side) and player (right side) of the screens
        court =obs[15:77, 12:71]
        player = obs[15:77, 72:76]

        #Locates the location of the ball using thresholds for the intensity then extracting the Y axis
        ball_pixels = np.argwhere((court > 0.4) & (court < 0.9))
        ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None



        #Locates the location of the player paddle using thresholds for the intensity then extracting the Y axis
        paddle_pixels = np.argwhere((player > 0.5) & (player < 0.7))
        paddle_y = float(np.mean(paddle_pixels[:, 0])) if len(paddle_pixels) > 0 else None
        return ball_y, paddle_y

    def picking(self):
        """
        Allowing the use of pretrained agents in the local PC
        """
        options = []
        for i in os.listdir():
            if os.path.exists(f"{i}/Checkpoints/brain4800.pth"):
                options.append(f"{i}/Checkpoints/brain4800.pth")

        print("Pick from the list which Agent you would like to evalute:")
        for i in range(len(options)):
            print(f"{i+1}.{options[i].split('/')[0]}")
        

        while True:
            try:
                choice = int(input()) - 1

                if choice > len(options) or choice < 0:
                    print("Your option doesn't exist in the list, please pick something from the list")
                    continue
                print("loading in ", options[choice])
                self.load_checkpoint(options[choice])
                self.agent = options[choice]
                break 

            except ValueError:
                print("Please enter a number")
                continue


class Network(nn.Module):
    def __init__(self, n_actions=6, dueling=False, noisy=False):
        super().__init__()
        self.dueling = dueling
        Linear = NoisyLinear if noisy else nn.Linear
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * 7 * 7
        if dueling:
            self.value = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, 1))
            self.advantage = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, n_actions))
        else:
            self.fc = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, n_actions))

    def forward(self, x):
        feats = self.conv(x)
        if self.dueling:
            v, a = self.value(feats), self.advantage(feats)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.fc(feats)
    
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

class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=0.5):
        super().__init__()
        bound = 1 / in_f ** 0.5
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((out_f, in_f), sigma_init / in_f ** 0.5))
        self.bias_mu = nn.Parameter(torch.empty(out_f).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(torch.full((out_f,), sigma_init / out_f ** 0.5))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, weight, bias)