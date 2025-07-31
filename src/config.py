"""
Configuration settings for JAX-based SAC training.
"""

from algorithms.sac import AutoAlphaConfig, SACConfig

# ----------------------------
# Algorithm & training config
# ----------------------------
# Environment settings
NUM_ENVS = 128
MAX_EPISODE_STEPS = 1000
USE_DOUBLE_PENDULUM = True  # Set to True for double pendulum, False for single pendulum

# Training settings
TOTAL_UPDATES = 200_000 if not USE_DOUBLE_PENDULUM else 1_000_000
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 256
UPDATES_PER_STEP = NUM_ENVS // 4  # network updates per env step
NETWORK_UPDATES_PER_GPU_CHUNK = 100  # updates per GPU-only chunk
STEPS_PER_GPU_CHUNK = (NETWORK_UPDATES_PER_GPU_CHUNK + UPDATES_PER_STEP - 1) // UPDATES_PER_STEP
EMA_BETA = 0.01  # smoothing for meters

SAC_CONFIG = SACConfig(
    learning_rate=3e-4,
    gamma=0.995,
    tau=0.005,
    alpha_config=AutoAlphaConfig(min_alpha=0.03),
    hidden_dims=(128, 128) if not USE_DOUBLE_PENDULUM else (128, 128, 128),
)


# Visualization settings
ENABLE_TRAINING_VIZ = True
ENABLE_LIVE_VIZ = True
