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

Transition = namedtuple('Transition', ['camera_state', 'joint_state', 'command', 'action', 'reward', 'camera_next', 'joint_next', 'done', 'n'])

ACTION_DIM   = 6
ACTION_SCALE = np.deg2rad(5.0)
LOG_STD_MIN  = -5
LOG_STD_MAX  = 2

class Brain:
    
    def __init__(self,ce=256, je=64, use_camera=True, multi_task=True, lr=config.LEARNING_RATE, wp=config.WARMUP, b=config.BATCH, g=config.GAMMA, tau=config.TAU,c=config.CAPACITY):


        self.use_camera = use_camera
        num_commands = 6 if multi_task else 1
        cam_dim = ce if use_camera else 0


        self.warmup  = wp
        self.batch   = b
        self.gamma   = g
        self.tau     = tau
        self.device  = "cuda"


        if use_camera:
            self.encoder   = CameraNetwork(cam_embedding_size=ce).to(self.device)
            self.target_encoder   = CameraNetwork(cam_embedding_size=ce).to(self.device)
            self.target_encoder.load_state_dict(self.encoder.state_dict())
        else:
            self.encoder = self.target_encoder = None


        self.joint_mlp = JointsNetwork(je).to(self.device)
        self.target_joint_mlp = JointsNetwork(je).to(self.device)


        self.target_joint_mlp.load_state_dict(self.joint_mlp.state_dict())

        # actor
        self.actor = Actor(cam_dim, je, num_commands).to(self.device)

        # twin critics
        self.critic        = Critic(cam_dim, je,num_commands).to(self.device)
        self.critic_target = Critic(cam_dim, je, num_commands).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # critic optimiser trains encoder + joint_mlp + critics
        critic_params = list(self.joint_mlp.parameters()) + list(self.critic.parameters())
        if use_camera:
            critic_params += list(self.encoder.parameters())
        self.critic_optimiser = torch.optim.Adam(critic_params, lr=lr)

        # actor optimiser does NOT include encoder or joint_mlp
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(), lr=lr,
        )

        # automatic entropy tuning
        self.target_entropy  = -13
        self.log_alpha = torch.tensor(np.log(0.05), requires_grad=True, device=self.device)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=lr)

        self.buffer       = ReplayBuffer(c)
        self.reward = 0
        self.critic_loss_ema = None
        self.critic_loss_ema_decay = 0.99
        self.alpha_warmup_value = 0.05
        self.critic_loss_warmup_ratio = 0.10  
        self.critic_loss_ema_initial = None
        self._warmup_baseline_samples = []



    def predict_next_action(
        self,
        camera_state: np.ndarray,   # [16, 224, 224]
        joint_state:  np.ndarray,   # [6]
        command: int,
        deterministic:    bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            cam_t = torch.FloatTensor(camera_state).unsqueeze(0).to(self.device) if self.use_camera else None
            joint_t = torch.FloatTensor(joint_state).unsqueeze(0).to(self.device)
            cmd_t   = torch.LongTensor([command]).to(self.device)

            cam_emb = self._get_cam_emb(cam_t, joint_t.shape[0])
            joint_emb = self.joint_mlp(joint_t).detach()

            if deterministic:
                action = self.actor.deterministic_action(cam_emb, joint_emb, cmd_t)
            else:
                action, _ = self.actor(cam_emb, joint_emb, cmd_t)
        return action.squeeze(0).cpu().numpy()

    def predict_next_action_batch(
    self,
    camera_states: np.ndarray,   # [N, 16, 224, 224]
    joint_states:  np.ndarray,   # [N, 6]
    commands: np.ndarray,
    deterministic: bool = False,
    ) -> np.ndarray:
        """
        Batched version of predict_next_action — runs one forward pass for
        all N envs instead of N separate calls. Returns [N, action_dim].
        """
        with torch.no_grad():
            cam_t = torch.FloatTensor(camera_states).to(self.device) if self.use_camera else None
            joint_t = torch.FloatTensor(joint_states).to(self.device)    # [N, 6]
            cmd_t   = torch.LongTensor(commands).to(self.device)

            cam_emb = self._get_cam_emb(cam_t, joint_t.shape[0])
            joint_emb = self.joint_mlp(joint_t).detach()

            if deterministic:
                fused  = torch.cat([cam_emb, joint_emb, cmd_t], dim=1)   # cmd_t is int64 concatenated with float32
                hidden = self.actor.net(fused)
                action = torch.tanh(self.actor.mean_head(hidden)) * ACTION_SCALE 
            else:
                action, _ = self.actor(cam_emb, joint_emb, cmd_t)

        return action.cpu().numpy()   # [N, action_dim]
 
    def train(self):
        if len(self.buffer) < self.warmup:
            return 0, 0, {}
        batch, indices, weights = self.buffer.sample(self.batch)
        weights = weights.to(self.device)

        joint_states = torch.FloatTensor(np.array([t.joint_state for t in batch])).to(self.device)
        commands     = torch.LongTensor(np.array([t.command       for t in batch])).to(self.device)
        actions      = torch.FloatTensor(np.array([t.action       for t in batch])).to(self.device)
        rewards      = torch.FloatTensor(np.array([t.reward       for t in batch])).to(self.device)
        joint_nexts  = torch.FloatTensor(np.array([t.joint_next   for t in batch])).to(self.device)
        dones        = torch.FloatTensor(np.array([t.done         for t in batch])).to(self.device)
        n_steps      = torch.FloatTensor(np.array([t.n            for t in batch])).to(self.device)

        if self.use_camera:
            cam_states = torch.FloatTensor(np.array([(t.camera_state.astype(np.float32) / 255.0) for t in batch])).to(self.device)
            cam_nexts  = torch.FloatTensor(np.array([(t.camera_next.astype(np.float32) / 255.0) for t in batch])).to(self.device)
        else:
            cam_states = cam_nexts = None   

        alpha_used = self._get_alpha_for_training()
        in_warmup = (self.critic_loss_ema_initial is None) or \
                    (self.critic_loss_ema >= self.critic_loss_ema_initial * self.critic_loss_warmup_ratio)

        # critic update
        with torch.no_grad():
            cam_next_emb   = self._get_cam_emb(cam_nexts, joint_states.shape[0], target=True)
            joint_next_emb = self.target_joint_mlp(joint_nexts)

            next_action, next_log_prob = self.actor(cam_next_emb, joint_next_emb, commands, freeze_std=in_warmup)
            q1_next, q2_next = self.critic_target(cam_next_emb, joint_next_emb, next_action, commands)
            q_next   = torch.min(q1_next, q2_next).squeeze(1)
            target_q = rewards + (self.gamma ** n_steps) * (1 - dones) * \
                    (q_next - alpha_used * next_log_prob)

        cam_emb   = self._get_cam_emb(cam_states, joint_states.shape[0])
        joint_emb = self.joint_mlp(joint_states)
        q1, q2    = self.critic(cam_emb, joint_emb, actions, commands)
        q1, q2    = q1.squeeze(1), q2.squeeze(1)

        td_errors   = (target_q - q1).detach().cpu().numpy()
        critic_loss = (
            weights * F.mse_loss(q1, target_q, reduction="none")
            + weights * F.mse_loss(q2, target_q, reduction="none")
        ).mean()

        cl = critic_loss.item()
        if self.critic_loss_ema_initial is None:
            self._warmup_baseline_samples.append(cl)
            if len(self._warmup_baseline_samples) >= 10:
                self.critic_loss_ema_initial = float(np.mean(self._warmup_baseline_samples))
                self.critic_loss_ema = self.critic_loss_ema_initial
        else:
            self.critic_loss_ema = self.critic_loss_ema_decay * self.critic_loss_ema + (1 - self.critic_loss_ema_decay) * cl

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_params = list(self.joint_mlp.parameters()) + list(self.critic.parameters())
        if self.use_camera:
            critic_params += list(self.encoder.parameters())
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, 10)
        self.critic_optimiser.step()

        # actor update — encoder gradients stopped
        cam_emb_d   = cam_emb.detach()
        joint_emb_d = joint_emb.detach()

        new_action, log_prob = self.actor(cam_emb_d, joint_emb_d, commands, freeze_std=in_warmup)
        q1_new, q2_new       = self.critic(cam_emb_d, joint_emb_d, new_action, commands)
        q_new      = torch.min(q1_new, q2_new).squeeze(1)
        actor_loss = (alpha_used * log_prob - q_new).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimiser.step()

        # entropy temperature update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.log_alpha.data.clamp_(max=np.log(5))

        # soft update
        if self.use_camera:
            self._soft_update(self.encoder, self.target_encoder)
        self._soft_update(self.joint_mlp, self.target_joint_mlp)
        self._soft_update(self.critic,    self.critic_target)

        with torch.no_grad():
            std_per_sample = self.actor.raw_std(cam_emb_d, joint_emb_d, commands)
            std_per_sample_mean = std_per_sample.mean(dim=-1)
            action_per_sample_mean = new_action.mean(dim=-1)

        num_commands = self.actor.num_commands
        per_cmd_std_log  = {f"train/action_std_cmd{c}":  std_per_sample_mean[commands == c].mean().item()
                            for c in range(num_commands) if (commands == c).any()}
        per_cmd_mean_log = {f"train/action_mean_cmd{c}": action_per_sample_mean[commands == c].mean().item()
                            for c in range(num_commands) if (commands == c).any()}

        diagnostics = {
            "train/q1_mean":          q1.mean().item(),
            "train/q2_mean":          q2.mean().item(),
            "train/q1_q2_gap":        (q1 - q2).abs().mean().item(),
            "train/target_q_mean":    target_q.mean().item(),
            "train/td_error_mean":    float(np.abs(td_errors).mean()),
            "train/log_prob_mean":    log_prob.mean().item(),
            "train/entropy":          -log_prob.mean().item(),
            "train/alpha_loss":       alpha_loss.item(),
            "train/action_mean":      new_action.mean().item(),
            "train/critic_grad_norm": critic_grad_norm.item(),
            "train/actor_grad_norm":  actor_grad_norm.item(),
            "train/target_q_std":     target_q.std().item(),
            "train/q1_std":           q1.std().item(),
            **per_cmd_std_log,
            **per_cmd_mean_log,
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
        ckpt = {
            "joint_mlp":        self.joint_mlp.state_dict(),
            "target_joint_mlp": self.target_joint_mlp.state_dict(),
            "actor":            self.actor.state_dict(),
            "critic":           self.critic.state_dict(),
            "critic_target":    self.critic_target.state_dict(),
            "critic_opt":       self.critic_optimiser.state_dict(),
            "actor_opt":        self.actor_optimiser.state_dict(),
            "alpha_opt":        self.alpha_optimiser.state_dict(),
            "log_alpha":        self.log_alpha.detach().cpu(),
            "episode":          episode,
            "steps":            steps,
            "use_camera":       self.use_camera,
            "num_commands":     self.actor.num_commands,
        }
        if self.use_camera:
            ckpt["encoder"]        = self.encoder.state_dict()
            ckpt["target_encoder"] = self.target_encoder.state_dict()
        torch.save(ckpt, os.path.join(path, f"manipulation_brain_{episode}.pth"))

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        ckpt = torch.load(path, map_location=self.device)

        ckpt_use_camera   = ckpt.get("use_camera", True)   
        ckpt_num_commands = ckpt.get("num_commands", 6)
        if ckpt_use_camera != self.use_camera:
            raise ValueError(
                f"checkpoint was saved with use_camera={ckpt_use_camera}, "
                f"but this Brain was built with use_camera={self.use_camera} — "
                f"architectures don't match, refusing to load."
            )
        if ckpt_num_commands != self.actor.num_commands:
            raise ValueError(
                f"checkpoint was saved with num_commands={ckpt_num_commands}, "
                f"but this Brain's actor has num_commands={self.actor.num_commands} — "
                f"architectures don't match, refusing to load."
            )

        if self.use_camera:
            self.encoder.load_state_dict(ckpt["encoder"])
            self.target_encoder.load_state_dict(ckpt["target_encoder"])

        self.joint_mlp.load_state_dict(ckpt["joint_mlp"])
        self.target_joint_mlp.load_state_dict(ckpt["target_joint_mlp"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

        self.critic_optimiser.load_state_dict(ckpt["critic_opt"])
        self.actor_optimiser.load_state_dict(ckpt["actor_opt"])
        self.alpha_optimiser.load_state_dict(ckpt["alpha_opt"])

        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        return ckpt.get("steps", 0), ckpt.get("episode", 0)
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _get_alpha_for_training(self) -> torch.Tensor:
        if self.critic_loss_ema is None or self.critic_loss_ema_initial is None:
            return torch.tensor(self.alpha_warmup_value, device=self.device)
        if self.critic_loss_ema < self.critic_loss_ema_initial * self.critic_loss_warmup_ratio:
            return self.alpha.detach()
        return torch.tensor(self.alpha_warmup_value, device=self.device)

    def _get_cam_emb(self, cam_t, batch_size: int, target: bool = False):

        if not self.use_camera:
            return torch.empty(batch_size, 0, device=self.device)
        enc = self.target_encoder if target else self.encoder
        return enc(cam_t)
        
# ADD Actor
class Actor(nn.Module):

    def __init__(self, cam_embedding_size: int = 256, joint_embedding_size: int = 64, num_commands=6):
        super().__init__()
        fused_size = cam_embedding_size + joint_embedding_size
        self.net   = nn.Sequential(
            nn.Linear(fused_size, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.mean_heads    = nn.ModuleList([nn.Linear(256, ACTION_DIM) for _ in range(num_commands)])
        self.log_std_heads = nn.ModuleList([nn.Linear(256, ACTION_DIM) for _ in range(num_commands)])
        for head in self.log_std_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, np.log(0.35))
        self.num_commands = num_commands

    def _select_head(self, all_outputs: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        
        idx = command.view(-1, 1, 1).expand(-1, 1, ACTION_DIM)
        return all_outputs.gather(1, idx).squeeze(1)

    def deterministic_action(self, cam_emb, joint_emb, command: torch.Tensor) -> torch.Tensor:
        fused  = torch.cat([cam_emb, joint_emb], dim=1)
        hidden = self.net(fused)
        all_means = torch.stack([h(hidden) for h in self.mean_heads], dim=1)
        mean = self._select_head(all_means, command)
        return torch.tanh(mean) * ACTION_SCALE

    def raw_std(self, cam_emb, joint_emb, command: torch.Tensor) -> torch.Tensor:
        """Deterministic per-sample std prediction — diagnostic only, no sampling noise."""
        fused  = torch.cat([cam_emb, joint_emb], dim=1)
        hidden = self.net(fused)
        all_log_stds = torch.stack([h(hidden) for h in self.log_std_heads], dim=1)
        log_std = self._select_head(all_log_stds, command).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return log_std.exp()

    def forward(self, cam_emb, joint_emb, command: torch.Tensor, freeze_std: bool = False):
        fused  = torch.cat([cam_emb, joint_emb], dim=1)
        hidden = self.net(fused)

        all_means    = torch.stack([h(hidden) for h in self.mean_heads], dim=1)
        mean = self._select_head(all_means, command)

        if freeze_std:
            std = torch.full_like(mean, 0.35)         
        else:
            all_log_stds = torch.stack([h(hidden) for h in self.log_std_heads], dim=1)
            log_std = self._select_head(all_log_stds, command).clamp(LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()

        dist   = torch.distributions.Normal(mean, std)
        x_t    = dist.rsample()
        tanh_x = torch.tanh(x_t)
        action = tanh_x * ACTION_SCALE

        tanh_x_safe = tanh_x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(x_t) - torch.log(ACTION_SCALE * (1 - tanh_x_safe.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

# ADD Critic
class Critic(nn.Module):
    def __init__(self, cam_embedding_size: int = 256, joint_embedding_size: int = 64, num_commands=6):
        super().__init__()
        fused_size = cam_embedding_size + joint_embedding_size  + ACTION_DIM
        self.q1_body = nn.Sequential(nn.Linear(fused_size, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.q2_body = nn.Sequential(nn.Linear(fused_size, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.q1_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_commands)])
        self.q2_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_commands)])

    def forward(self, cam_emb, joint_emb, action, command: torch.Tensor):
        fused = torch.cat([cam_emb, joint_emb, action], dim=1)
        h1, h2 = self.q1_body(fused), self.q2_body(fused)

        all_q1 = torch.stack([h(h1) for h in self.q1_heads], dim=1)  
        all_q2 = torch.stack([h(h2) for h in self.q2_heads], dim=1)

        idx = command.view(-1, 1, 1)
        q1 = all_q1.gather(1, idx).squeeze(1)
        q2 = all_q2.gather(1, idx).squeeze(1)
        return q1, q2

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


class CameraNetwork_Resnet(nn.Module):
    #heavier approach which builds from a resnet dropped currently due to hardware limitation and it is not needed for the current approach

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


class CameraNetwork(nn.Module):

    def __init__(self, in_channels: int = 12, cam_embedding_size: int = 256):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel, stride, padding, groups=8):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                          padding=padding, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_block(in_channels, 32, kernel=4, stride=2, padding=1),  # 128 -> 64
            conv_block(32, 64, kernel=4, stride=2, padding=1),           # 64  -> 32
            conv_block(64, 64, kernel=4, stride=2, padding=1),           # 32  -> 16
            conv_block(64, 64, kernel=3, stride=2, padding=1),           # 16  -> 8
            conv_block(64, 64, kernel=3, stride=2, padding=1),           # 8   -> 4
        )

        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, cam_embedding_size),
            nn.LayerNorm(cam_embedding_size),
            nn.ReLU(),
        )

        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_head(self.backbone(x))


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