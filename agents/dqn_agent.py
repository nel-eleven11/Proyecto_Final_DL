# agents/dqn_agent.py
from __future__ import annotations
from stable_baselines3 import DQN

def crear_dqn(env, tensorboard_log: str | None = "logs/tb", lr: float = 2.5e-4,
              buffer_size: int = 100_000, batch_size: int = 64, gamma: float = 0.99,
              train_freq: int = 4, target_update_interval: int = 1000,
              exploration_fraction: float = 0.3, exploration_final_eps: float = 0.05) -> DQN:
    """Crea un agente DQN con política CNN (para patch egocéntrico)."""
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
    )
    return model