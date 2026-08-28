import numpy as np 
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from random import random
import torch.optim as optim
import torchvision.models as models

import ale.config as config

Transition = namedtuple('Transition', ['camera_state', 'joint_state', 'action', 'reward', 'camera_next', 'joint_next', 'done', 'n'])

ACTION_DIM   = 6
ACTION_SCALE = np.deg2rad(5.0)
LOG_STD_MIN  = -5
LOG_STD_MAX  = 2

class Brain:
    
    def __init__(self,ce=256, je=64, lr=config.LEARNING_RATE, wp=config.WARMUP, b=config.BATCH, g=config.GAMMA, tau=config.TAU,c=config.CAPACITY):
        
        self.warmup  = wp
        self.batch   = b
        self.gamma   = g
        self.tau     = tau
        self.device  = "cuda"

        # shared encoder and joint mlp
        self.encoder   = CameraNetwork(ce).to(self.device)
        self.joint_mlp = JointsNetwork(je).to(self.device)

        # target encoder and joint mlp
        self.target_encoder   = CameraNetwork(ce).to(self.device)
        self.target_joint_mlp = JointsNetwork(je).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_joint_mlp.load_state_dict(self.joint_mlp.state_dict())

        # actor
        self.actor = Actor(ce, je).to(self.device)

        # twin critics
        self.critic        = Critic(ce, je).to(self.device)
        self.critic_target = Critic(ce, je).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # critic optimiser trains encoder + joint_mlp + critics
        self.critic_optimiser = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.joint_mlp.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )
        # actor optimiser does NOT include encoder or joint_mlp
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(), lr=lr,
        )

        # automatic entropy tuning
        self.target_entropy  = -ACTION_DIM
        self.log_alpha       = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=lr)

        self.buffer       = ReplayBuffer(c)
        self.reward = RunningNormaliser()
        # self.success_count = 0
        # self.min_successes_before_decay = 10  
        # self.alpha_floor = 0.01



    def predict_next_action(
        self,
        camera_state: np.ndarray,   # [16, 224, 224]
        joint_state:  np.ndarray,   # [6]
        deterministic:    bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            cam_t   = torch.FloatTensor(camera_state).unsqueeze(0).to(self.device)
            joint_t = torch.FloatTensor(joint_state).unsqueeze(0).to(self.device)
            cam_emb   = self.encoder(cam_t).detach()
            joint_emb = self.joint_mlp(joint_t).detach()
            if deterministic:
                fused  = torch.cat([cam_emb, joint_emb], dim=1)
                hidden = self.actor.net(fused)
                action = torch.tanh(self.actor.mean_head(hidden)) * ACTION_SCALE
            else:
                action, _ = self.actor(cam_emb, joint_emb)
        return action.squeeze(0).cpu().numpy()

    def predict_next_action_batch(
    self,
    camera_states: np.ndarray,   # [N, 16, 224, 224]
    joint_states:  np.ndarray,   # [N, 6]
    deterministic: bool = False,
    ) -> np.ndarray:
        """
        Batched version of predict_next_action — runs one forward pass for
        all N envs instead of N separate calls. Returns [N, action_dim].
        """
        with torch.no_grad():
            cam_t   = torch.FloatTensor(camera_states).to(self.device)   # [N, 16, 224, 224]
            joint_t = torch.FloatTensor(joint_states).to(self.device)    # [N, 6]

            cam_emb   = self.encoder(cam_t).detach()
            joint_emb = self.joint_mlp(joint_t).detach()

            if deterministic:
                fused  = torch.cat([cam_emb, joint_emb], dim=1)
                hidden = self.actor.net(fused)
                action = torch.tanh(self.actor.mean_head(hidden)) * ACTION_SCALE
            else:
                action, _ = self.actor(cam_emb, joint_emb)

        return action.cpu().numpy()   # [N, action_dim]
 
    def train(self):

        if len(self.buffer) < self.warmup:
            return 0, 0, {}

        batch, indices, weights = self.buffer.sample(self.batch)
        weights = weights.to(self.device)

        cam_states   = torch.FloatTensor(np.array([(t.camera_state.astype(np.float32) / 255.0) for t in batch])).to(self.device)
        joint_states = torch.FloatTensor(np.array([t.joint_state  for t in batch])).to(self.device)
        actions      = torch.FloatTensor(np.array([t.action       for t in batch])).to(self.device)
        rewards      = torch.FloatTensor(np.array([t.reward       for t in batch])).to(self.device)
        cam_nexts    = torch.FloatTensor(np.array([(t.camera_next.astype(np.float32) / 255.0) for t in batch])).to(self.device)
        joint_nexts  = torch.FloatTensor(np.array([t.joint_next   for t in batch])).to(self.device)
        dones        = torch.FloatTensor(np.array([t.done         for t in batch])).to(self.device)
        n_steps      = torch.FloatTensor(np.array([t.n            for t in batch])).to(self.device)

        # critic update
        with torch.no_grad():
            cam_next_emb   = self.target_encoder(cam_nexts)
            joint_next_emb = self.target_joint_mlp(joint_nexts)
            next_action, next_log_prob = self.actor(cam_next_emb, joint_next_emb)
            q1_next, q2_next = self.critic_target(cam_next_emb, joint_next_emb, next_action)
            q_next   = torch.min(q1_next, q2_next).squeeze(1)
            target_q = rewards + (self.gamma ** n_steps) * (1 - dones) * \
                    (q_next - self.alpha.detach() * next_log_prob)

        cam_emb   = self.encoder(cam_states)
        joint_emb = self.joint_mlp(joint_states)
        q1, q2    = self.critic(cam_emb, joint_emb, actions)
        q1, q2    = q1.squeeze(1), q2.squeeze(1)

        td_errors   = (target_q - q1).detach().cpu().numpy()
        critic_loss = (
            weights * F.mse_loss(q1, target_q, reduction="none")
            + weights * F.mse_loss(q2, target_q, reduction="none")
        ).mean()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.joint_mlp.parameters())
            + list(self.critic.parameters()), 10
        )
        self.critic_optimiser.step()
        self.buffer.update_priorities(indices, td_errors)

        # actor update — encoder gradients stopped
        cam_emb_d   = cam_emb.detach()
        joint_emb_d = joint_emb.detach()
        new_action, log_prob = self.actor(cam_emb_d, joint_emb_d)
        q1_new, q2_new       = self.critic(cam_emb_d, joint_emb_d, new_action)
        q_new      = torch.min(q1_new, q2_new).squeeze(1)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimiser.step()

        # entropy temperature update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        
        # soft update
        self._soft_update(self.encoder,   self.target_encoder)
        self._soft_update(self.joint_mlp, self.target_joint_mlp)
        self._soft_update(self.critic,    self.critic_target)

        n = len(self.buffer)
        diagnostics = {
            "train/q1_mean":         q1.mean().item(),
            "train/q2_mean":         q2.mean().item(),
            "train/q1_q2_gap":       (q1 - q2).abs().mean().item(),
            "train/target_q_mean":   target_q.mean().item(),
            "train/td_error_mean":   float(np.abs(td_errors).mean()),
            "train/log_prob_mean":   log_prob.mean().item(),
            "train/entropy":         -log_prob.mean().item(),
            "train/alpha_loss":      alpha_loss.item(),
            "train/action_mean":     new_action.mean().item(),
            "train/action_std":      new_action.std().item(),
            "train/critic_grad_norm": critic_grad_norm.item(),
            "train/actor_grad_norm":  actor_grad_norm.item(),
            "train/priority_mean":   float(self.buffer.priorities[:n].mean()),
            "train/priority_max":    float(self.buffer.priorities[:n].max()),
        }

        return critic_loss.item(), actor_loss.item(), diagnostics
 
    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
 
    def save_checkpoint(
        self,
        episode: int,
        steps:   int,
        path:    str,
    ):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "encoder":          self.encoder.state_dict(),
            "joint_mlp":        self.joint_mlp.state_dict(),
            "target_encoder":   self.target_encoder.state_dict(),
            "target_joint_mlp": self.target_joint_mlp.state_dict(),
            "actor":            self.actor.state_dict(),
            "critic":           self.critic.state_dict(),
            "critic_target":    self.critic_target.state_dict(),
            "critic_opt":       self.critic_optimiser.state_dict(),
            "actor_opt":        self.actor_optimiser.state_dict(),
            "alpha_opt":        self.alpha_optimiser.state_dict(),
            "log_alpha":        self.log_alpha.detach().cpu(),
            "episode":      episode,
            "steps":        steps,
            "norm_success": self.reward.state_dict(),
        }, os.path.join(path, f"manipulation_brain_{episode}.pth"))
 
    def load_checkpoint(self, path: str) -> tuple[int, int]:
        ckpt = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.joint_mlp.load_state_dict(ckpt["joint_mlp"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.target_joint_mlp.load_state_dict(ckpt["target_joint_mlp"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

        self.critic_optimiser.load_state_dict(ckpt["critic_opt"])
        self.actor_optimiser.load_state_dict(ckpt["actor_opt"])
        self.alpha_optimiser.load_state_dict(ckpt["alpha_opt"])

        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        if "norm_success" in ckpt:
            self.reward.load_state_dict(ckpt["norm_success"])

        return ckpt.get("steps", 0), ckpt.get("episode", 0)
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
# ADD Actor
class Actor(nn.Module):

    def __init__(self, cam_embedding_size: int = 256, joint_embedding_size: int = 64):
        super().__init__()
        fused_size = cam_embedding_size + joint_embedding_size
        self.net   = nn.Sequential(
            nn.Linear(fused_size, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, ACTION_DIM)
        self.log_std_head = nn.Linear(256, ACTION_DIM)

    def forward(self, cam_emb, joint_emb):
        fused   = torch.cat([cam_emb, joint_emb], dim=1)
        hidden  = self.net(fused)
        mean    = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mean, std)
        x_t     = dist.rsample()
        tanh_x  = torch.tanh(x_t)
        action  = tanh_x * ACTION_SCALE

        log_prob = dist.log_prob(x_t) - torch.log(1 - tanh_x.pow(2) + 1e-6)  
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


# ADD Critic
class Critic(nn.Module):
    def __init__(self, cam_embedding_size: int = 256, joint_embedding_size: int = 64):
        super().__init__()
        fused_size = cam_embedding_size + joint_embedding_size + ACTION_DIM
        self.q1 = nn.Sequential(
            nn.Linear(fused_size, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(fused_size, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, cam_emb, joint_emb, action):
        fused = torch.cat([cam_emb, joint_emb, action], dim=1)
        return self.q1(fused), self.q2(fused)

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
        orig = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=16, out_channels=orig.out_channels,
            kernel_size=orig.kernel_size, stride=orig.stride,
            padding=orig.padding, bias=False,
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cam_embedding_size),
            nn.LayerNorm(cam_embedding_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_head(self.backbone(x))


class RunningNormaliser:
    

    def __init__(self, epsilon = 1e-8):
        
        self.mean = 0
        self.var = 0.0
        self.count = 0

        self.epsilon = epsilon


    def update(self, value):
        self.count += 1

        delta = value - self.mean
        self.mean += delta/self.count
        self.var = self.var + delta * (value - self.mean)

    def normalise(self, value):
        std = np.sqrt(self.var/ max(self.count,1)) + self.epsilon
        norm = value/std
        print(norm)
        return np.clip(norm, -1, 1)
    

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
    Prioritised experience replay buffer.
    Samples transitions proportional to TD error so rare high-reward
    transitions are visited more frequently than zero-reward ones.
    """
 
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, use_priority=False):

        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta
        self.buffer    = []
        self.priorities = np.zeros(int(capacity), dtype=np.float32)
        self.pos       = 0
        self.use_priority = use_priority

 
    def push(self, *args):
        max_priority = self.priorities.max() if self.buffer else 1.0
 
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
 
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
 
    def sample(self, batch_size: int):
        n          = len(self.buffer)

        if not self.use_priority:
            indices = np.random.choice(n, batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
            weights = torch.ones(batch_size)  # uniform — no correction needed
            return samples, indices, weights
        
        priorities = self.priorities[:n]
        probs      = priorities ** self.alpha
        probs     /= probs.sum()
 
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
 
        weights  = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
 
        return samples, indices, torch.FloatTensor(weights)
 
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + 1e-6
 
    def __len__(self) -> int:
        return len(self.buffer)