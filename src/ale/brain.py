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
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Brain():
    """
    The class used for the RL agent
    """
    def __init__(self, lr=0, wp=0, b=0, g=0, tau=0, ee=0, es=0, ed=0,c=0):
        
        #initializing the policy network
        self.policy = Network().to("cuda")
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        #initializing the test network
        self.test = Network().to("cuda")
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

        self.agent = ""
    def train(self):
       """
       Training the policy netwrok, given the collected observations
       """

        #The warmup to allow the buffer to collect data before training starts
       if len(self.buffer) < self.warmup:
           return 0, 0
    
        #sampling different observations from the collected data
       batch = self.buffer.sample(self.batch)
       states = torch.FloatTensor(np.array([t.state      for t in batch])).to("cuda")
       actions = torch.LongTensor(np.array([t.action      for t in batch])).to("cuda")
       rewards = torch.FloatTensor(np.array([t.reward      for t in batch])).to("cuda")
       next_states = torch.FloatTensor(np.array([t.next_state      for t in batch])).to("cuda")
       dones = torch.FloatTensor(np.array([t.done      for t in batch])).to("cuda")
       
       #calculating the Q_Values of the collected states
       q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
       with torch.no_grad():
           #calculating the approximate next Q_values and the target
           next_q = self.test(next_states).max(1)[0]
           targets = rewards +self.gamma* next_q * (1 - dones)


        #calculating the loss and passing it backward
       loss = self.loss_fn(q_values, targets)
       self.optimiser.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
       self.optimiser.step()
       self.soft_update()

        #calculating the grad_norm for measuring the overall performace 
       grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in self.policy.parameters()
        if p.grad is not None
        ) ** 0.5


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

    def predict_next_action(self, state, steps, env):
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
            }, f"{path}/Checkpoints/brain{episode}.pth"
        )
    def save(self, path):
        """
        Save the final training step for furtherr eval 
        """
        torch.save(self.policy.state_dict(), f"{path}/brain.pth")

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
        court =obs[14:76, 16:79]
        player = obs[14:76, 70:79]

        #Locates the location of the ball using thresholds for the intensity then extracting the Y axis
        ball_pixels = np.argwhere((court > 0.7) & (court < 0.9))
        ball_y = float(np.mean(ball_pixels[:, 0])) if len(ball_pixels) > 0 else None

        #Locates the location of the player paddle using thresholds for the intensity then extracting the Y axis
        paddle_pixels = np.argwhere((player > 0.4) & (player < 0.9))
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
    """
    The CNN used in the training
    """
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

    """
    Used to save observations for the CNN and then sampled from for training purposes 
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    