# agents/dqn_agent.py
from __future__ import annotations
from typing import Dict, Any
import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN6CExtractor(BaseFeaturesExtractor):
    """Extractor CNN robusto para HxWxC (C puede ser !=6; usamos shape del espacio)."""
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        h, w, c = observation_space.shape  # acepta 8 canales (con S y M)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # -> (N, 64)
        )
        self.linear = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.permute(0, 3, 1, 2).contiguous()  # (N,H,W,C)->(N,C,H,W)
        x = self.cnn(x)
        return self.linear(x)

def crear_dqn(
    env,
    tensorboard_log: str | None = "logs/tb",
    lr: float = 2.5e-4,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    train_freq: int = 4,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    verbose: int = 1,              
) -> DQN:
    """Crea un DQN con política CNN y extractor personalizado (acepta 8 canales)."""
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=CNN6CExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )
    return DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=verbose,                 # << configurable
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )
