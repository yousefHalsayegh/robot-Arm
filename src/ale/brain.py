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
    def __init__(self, lr=0, wp=0, b=0, g=0, tau=0, ee=0, es=0, ed=0,c=0, d=False, tu="soft", tup=8000, n=False, distributional=False, n_atoms=51,
             v_min=-1.0, v_max=1.0):
        
        self.distributional = distributional
        self.n_atoms = n_atoms
        self.v_min, self.v_max = v_min, v_max

        self.policy = Network(dueling=d, noisy=n, distributional=distributional,
                            n_atoms=n_atoms, v_min=v_min, v_max=v_max).to("cuda")
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr, eps=1.5e-4)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')   # unused when distributional, kept for the scalar path

        self.test = Network(dueling=d, noisy=n, distributional=distributional,
                            n_atoms=n_atoms, v_min=v_min, v_max=v_max).to("cuda")
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
        if len(self.buffer) < self.warmup:
            return 0, 0

        batch, indices, weights = self.buffer.sample(self.batch)

        states      = torch.from_numpy(batch["states"]).to("cuda").float().div_(255.0)
        actions     = torch.from_numpy(batch["actions"]).to("cuda")
        rewards     = torch.from_numpy(batch["rewards"]).to("cuda")
        next_states = torch.from_numpy(batch["next_states"]).to("cuda").float().div_(255.0)
        dones       = torch.from_numpy(batch["dones"]).to("cuda")
        steps       = torch.from_numpy(batch["ns"]).to("cuda")

        if self.distributional:
            n_atoms = self.n_atoms
            support = self.policy.support

            with torch.no_grad():
                next_q_policy = self._q_values(self.policy, next_states)          
                next_actions = next_q_policy.argmax(dim=1)

                next_log_probs_target = self.test(next_states)
                next_probs_target = next_log_probs_target.exp()
                next_dist = next_probs_target.gather(
                    1, next_actions.view(-1, 1, 1).expand(-1, 1, n_atoms)
                ).squeeze(1)                                                       

                gamma_n = self.gamma ** steps
                target_dist = self.project_distribution(next_dist, rewards, dones, gamma_n,
                                                    support, self.v_min, self.v_max, n_atoms)

            log_probs = self.policy(states)
            log_probs_a = log_probs.gather(
                1, actions.view(-1, 1, 1).expand(-1, 1, n_atoms)
            ).squeeze(1)                                                           

            per_sample_loss = -(target_dist * log_probs_a).sum(dim=1)              
            loss = (weights * per_sample_loss).mean()
            td_errors = per_sample_loss.detach().cpu().numpy()                     
            self.q_value = self._q_values(self.policy, states).gather(1, actions.unsqueeze(1)).squeeze(1)

        else:
            # --- original scalar path, unchanged ---
            q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            self.q_value = q_values
            with torch.no_grad():
                next_actions = self.policy(next_states).argmax(1, keepdim=True)
                next_q = self.test(next_states).gather(1, next_actions).squeeze(1)
                targets = rewards + (self.gamma ** steps) * next_q * (1 - dones)
            per_sample_loss = self.loss_fn(q_values, targets)
            loss = (weights * per_sample_loss).mean()
            td_errors = (targets - q_values).detach().cpu().numpy()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10).item()
        self.optimiser.step()
        self.train_steps += 1

        if self.target_update == "soft":
            self.soft_update()
        elif self.train_steps % self.target_update_period == 0:
            self.hard_update()

        self.buffer.update_priorities(indices, td_errors)
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
            greedy_action = self._q_values(self.policy, state_t).argmax(dim=1).cpu().numpy()

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
            return self._q_values(self.policy, state_next).argmax(dim=1).item()
        
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

    def project_distribution(self, next_probs, rewards, dones, gamma_n, support, v_min, v_max, n_atoms):
        delta_z = (v_max - v_min) / (n_atoms - 1)
        batch_size = rewards.shape[0]

        Tz = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        Tz = Tz.clamp(v_min, v_max)
        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l[(u > 0) & (l == u)] -= 1
        u[(l < n_atoms - 1) & (l == u)] += 1

        m = torch.zeros(batch_size, n_atoms, device=rewards.device)
        offset = (torch.arange(batch_size, device=rewards.device) * n_atoms).unsqueeze(1).expand(batch_size, n_atoms)
        m.view(-1).index_add_(0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))
        return m

    
    def _q_values(self, net, x):
        out = net(x)
        if self.distributional:
            return (out.exp() * net.support).sum(dim=2)
        return out

class Network(nn.Module):
    def __init__(self, n_actions=6, dueling=False, noisy=False,
                 distributional=False, n_atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.dueling = dueling
        self.distributional = distributional
        self.n_actions = n_actions
        self.n_atoms = n_atoms if distributional else 1

        if distributional:
            self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        Linear = NoisyLinear if noisy else nn.Linear
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * 7 * 7
        out_per_action = self.n_atoms 

        if dueling:
            self.value = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, out_per_action))
            self.advantage = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, n_actions * out_per_action))
        else:
            self.fc = nn.Sequential(Linear(conv_out, 512), nn.ReLU(), Linear(512, n_actions * out_per_action))

    def forward(self, x):
        feats = self.conv(x)
        B = x.shape[0]

        if self.distributional:
            if self.dueling:
                v = self.value(feats).view(B, 1, self.n_atoms)
                a = self.advantage(feats).view(B, self.n_actions, self.n_atoms)
                logits = v + (a - a.mean(dim=1, keepdim=True))
            else:
                logits = self.fc(feats).view(B, self.n_actions, self.n_atoms)
            return torch.log_softmax(logits, dim=2)   # [B, n_actions, n_atoms] log-probs
        else:
            if self.dueling:
                v, a = self.value(feats), self.advantage(feats)
                return v + (a - a.mean(dim=1, keepdim=True))
            return self.fc(feats)
    
class ReplayBuffer:

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = int(capacity)
        self.alpha    = alpha
        self.beta     = beta
        self.pos      = 0
        self.size     = 0         
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        # Arrays are allocated lazily on the first push(), once we know the
        # actual state shape — avoids hardcoding (4, 84, 84) here and needing
        # every existing Brain(...) call site to be changed.
        self._arrays_ready = False

    def _allocate(self, state_shape, action, reward, done, n):
        self.states      = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.actions     = np.zeros(self.capacity, dtype=np.int64)
        self.rewards     = np.zeros(self.capacity, dtype=np.float32)
        self.dones       = np.zeros(self.capacity, dtype=np.float32)
        self.ns          = np.zeros(self.capacity, dtype=np.float32)
        self._arrays_ready = True

    def push(self, state, action, reward, next_state, done, n):
        if not self._arrays_ready:
            self._allocate(state.shape, action, reward, done, n)

        # new transitions get max priority so they are sampled at least once
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.states[self.pos]      = state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos]       = float(done)
        self.ns[self.pos]          = n

        self.priorities[self.pos] = max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        n = self.size
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        # importance sampling weights — correct for sampling bias
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = {
            "states":      self.states[indices],       # already uint8 [B, 4, 84, 84]
            "actions":     self.actions[indices],
            "rewards":     self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones":       self.dones[indices],
            "ns":          self.ns[indices],
        }
        return batch, indices, torch.FloatTensor(weights).to("cuda")

    def update_priorities(self, indices, td_errors):
        """Call after train() with the computed TD errors."""
        self.priorities[indices] = np.abs(td_errors) + 1e-6   # vectorized, was a Python for-loop

    def __len__(self):
        return self.size

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