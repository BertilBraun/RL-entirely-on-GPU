"""
Configuration settings for JAX-based RL training.
"""

import jax
import jax.numpy as jnp
from training.data_structures import AutoAlphaConfig, SACConfig, PPOConfig
from typing import Union
from algorithms.base import Algorithm

# ----------------------------
# Algorithm Selection
# ----------------------------
ALGORITHM = 'PPO'  # "SAC" or "PPO"

# ----------------------------
# Algorithm & training config
# ----------------------------
# Environment settings
NUM_ENVS = 512
USE_DOUBLE_PENDULUM = True  # Set to True for double pendulum, False for single pendulum
MAX_EPISODE_STEPS = 1000 if not USE_DOUBLE_PENDULUM else 4000

# Training settings
TOTAL_UPDATES = 200_000 if not USE_DOUBLE_PENDULUM else 2_000_000
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 1024
UPDATES_PER_STEP = NUM_ENVS // 32  # network updates per env step
NETWORK_UPDATES_PER_GPU_CHUNK = 1000  # updates per GPU-only chunk
STEPS_PER_GPU_CHUNK = (NETWORK_UPDATES_PER_GPU_CHUNK + UPDATES_PER_STEP - 1) // UPDATES_PER_STEP
EMA_BETA = 0.01  # smoothing for meters

if USE_DOUBLE_PENDULUM:
    UPDATES_PER_STEP = 4  # TODO
    SAC_CONFIG = SACConfig(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        grad_clip=5.0,
        target_entropy=-1.5,
        alpha_config=AutoAlphaConfig(min_alpha=0.005),
        actor_hidden_dims=(64, 64, 64),
        critic_hidden_dims=(64, 64, 64, 64),
    )
    PPO_CONFIG = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        num_minibatches=4,
        actor_hidden_dims=(64, 64, 64),
        critic_hidden_dims=(64, 64, 64),
        normalize_advantages=True,
    )
    jax.config.update('jax_enable_x64', True)
    DTYPE = jnp.float32
else:
    SAC_CONFIG = SACConfig(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.003,
        grad_clip=5.0,
        target_entropy=None,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    PPO_CONFIG = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        num_minibatches=4,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        normalize_advantages=True,
    )
    DTYPE = jnp.float32

# Visualization settings
ENABLE_TRAINING_VIZ = True
ENABLE_LIVE_VIZ = True


def create_algorithm(obs_dim: int, action_dim: int, max_action: float) -> Algorithm:
    """
    Factory function to create the selected algorithm.

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        max_action: Maximum action value

    Returns:
        Configured algorithm instance (SAC or PPO)
    """
    if ALGORITHM == 'SAC':
        from algorithms.sac import SAC

        return SAC(obs_dim, action_dim, max_action, SAC_CONFIG)
    elif ALGORITHM == 'PPO':
        from algorithms.ppo import PPO

        return PPO(obs_dim, action_dim, max_action, PPO_CONFIG)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}. Choose 'SAC' or 'PPO'.")


def get_algorithm_config() -> Union[SACConfig, PPOConfig]:
    """Get the configuration for the selected algorithm."""
    if ALGORITHM == 'SAC':
        return SAC_CONFIG
    elif ALGORITHM == 'PPO':
        return PPO_CONFIG
    else:
        raise ValueError(f'Unknown algorithm: {ALGORITHM}')
