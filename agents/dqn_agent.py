# agents/dqn_agent.py
from __future__ import annotations
from typing import Dict, Any
import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def crear_dqn2(
    env, 
    tensorboard_log: str | None = "logs/tb", 
    lr: float = 2.5e-4,
    buffer_size: int = 100_000, 
    batch_size: int = 64, 
    gamma: float = 0.99,
    train_freq: int = 4, 
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.3, 
    exploration_final_eps: float = 0.05
) -> DQN:
    """Crea un agente DQN con política CNN (para patch egocéntrico)."""
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model

class CNN6CExtractor(BaseFeaturesExtractor):
    """
    Extractor CNN para observaciones HxWxC (C=6). Robusto a tamaños pequeños (11x11)
    gracias a kernels 3x3 + AdaptiveAvgPool2d(1).
    """
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Se espera Box(H, W, C)"
        h, w, c = observation_space.shape

        self.cnn = nn.Sequential(
            # Entrada: (N, C, H, W)
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Pooling adaptable a 1x1 para no depender del tamaño espacial
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> (N, 64)
        )

        # Con pooling a 1x1, el flatten queda fijo en 64
        n_flatten = 64

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs llega como (N, H, W, C) -> pasamos a (N, C, H, W)
        x = obs.permute(0, 3, 1, 2).contiguous()
        x = self.cnn(x)
        x = self.linear(x)
        return x

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
    exploration_final_eps: float = 0.05
) -> DQN:
    """
    Crea un DQN con política CNN y extractor personalizado (6 canales HxWxC).
    """
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=CNN6CExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,  # ya vienen en [0,1]
    )

    model = DQN(
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
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )
    return model