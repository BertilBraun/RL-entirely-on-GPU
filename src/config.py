"""
Configuration settings for JAX-based SAC training.
"""

import jax
import jax.numpy as jnp
from training.data_structures import AutoAlphaConfig, SACConfig

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
UPDATES_PER_STEP = NUM_ENVS // 64  # network updates per env step
NETWORK_UPDATES_PER_GPU_CHUNK = 1000  # updates per GPU-only chunk
STEPS_PER_GPU_CHUNK = (NETWORK_UPDATES_PER_GPU_CHUNK + UPDATES_PER_STEP - 1) // UPDATES_PER_STEP
EMA_BETA = 0.01  # smoothing for meters

if USE_DOUBLE_PENDULUM:
    SAC_CONFIG = SACConfig(
        learning_rate=3e-4,
        gamma=0.999,
        tau=0.005,
        grad_clip=10.0,
        target_entropy=-1.5,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256, 256),
    )
    jax.config.update('jax_enable_x64', True)
    DTYPE = jnp.float32
else:
    SAC_CONFIG = SACConfig(
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.003,
        grad_clip=10.0,
        target_entropy=None,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    DTYPE = jnp.float32

# Visualization settings
ENABLE_TRAINING_VIZ = True
ENABLE_LIVE_VIZ = True
