"""
Data structures for JAX-based RL training.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from algorithms.replay_buffer import ReplayBufferState
from algorithms.episode_buffer import EpisodeBufferState
from typing import TYPE_CHECKING, NamedTuple, Tuple, Union
from environment.cartpole import CartPoleEnv, CartPoleState
from environment.double_pendulum_cartpole import DoublePendulumCartPoleEnv, DoublePendulumCartPoleState

if TYPE_CHECKING:
    from algorithms.base import Algorithm, AlgorithmState

# ----------------------------
# Setup data structures
# ----------------------------


EnvType = DoublePendulumCartPoleEnv | CartPoleEnv
BufferStateType = Union[ReplayBufferState, EpisodeBufferState]


@chex.dataclass
class TrainingSetup:
    """Contains all components needed to initialize training."""

    env: EnvType
    algorithm: Algorithm
    algorithm_state: AlgorithmState
    buffer_state: BufferStateType
    initial_obs: chex.Array
    initial_env_state: Union[CartPoleState, DoublePendulumCartPoleState]
    rng: chex.PRNGKey


class AutoAlphaConfig(NamedTuple):
    """Configuration for auto alpha."""

    min_alpha: float = 0.03


class ManualAlphaConfig(NamedTuple):
    """Configuration for manual alpha."""

    alpha: float = 0.2


class SACConfig(NamedTuple):
    """Configuration for SAC algorithm."""

    # Learning rate for actor and critic networks
    learning_rate: float = 3e-4
    # Discount factor for future rewards
    gamma: float = 0.99
    # Soft update factor for target networks (how much of the new target network is mixed into the current target network)
    tau: float = 0.005
    # Gradient clipping for actor and critic networks
    grad_clip: float = 10.0
    # Configuration for alpha (temperature parameter for entropy regularization)
    alpha_config: AutoAlphaConfig | ManualAlphaConfig = AutoAlphaConfig()
    # Target entropy for automatic alpha tuning (None for -action_dim)
    target_entropy: float | None = None
    # Hidden dimensions for actor and critic networks
    actor_hidden_dims: Tuple[int, ...] = (32, 32)
    critic_hidden_dims: Tuple[int, ...] = (32, 32)


class PPOConfig(NamedTuple):
    """Configuration for PPO algorithm."""

    # Learning rate for actor and critic networks
    learning_rate: float = 3e-4
    # Discount factor for future rewards
    gamma: float = 0.99
    # GAE lambda for advantage estimation
    gae_lambda: float = 0.95
    # PPO clipping ratio
    clip_ratio: float = 0.2
    # Value function loss coefficient
    value_loss_coef: float = 0.5
    # Entropy bonus coefficient
    entropy_coef: float = 0.01
    # Maximum gradient norm for clipping
    max_grad_norm: float = 0.5
    # Number of PPO epochs per update
    ppo_epochs: int = 4
    # Number of minibatches to split episodes into during update
    num_minibatches: int = 4
    # Hidden dimensions for actor and critic networks
    actor_hidden_dims: Tuple[int, ...] = (64, 64)
    critic_hidden_dims: Tuple[int, ...] = (64, 64)
    # Whether to normalize advantages
    normalize_advantages: bool = True


# ----------------------------
# Training data structures
# ----------------------------


@chex.dataclass
class TrainCarry:
    """State that persists across chunks (host <-> device boundary)."""

    rng: chex.PRNGKey
    algorithm_state: AlgorithmState
    buffer_state: BufferStateType
    env_state: Union[CartPoleState, DoublePendulumCartPoleState]
    obs: chex.Array  # (num_envs, obs_dim)
    env_steps: chex.Array  # (num_envs,) int32
    episode_rewards: chex.Array  # (num_envs,) float32
    total_updates_done: chex.Array  # () int32

    @staticmethod
    def init(setup: TrainingSetup) -> TrainCarry:
        from config import DTYPE
        import jax.numpy as jnp

        # Get actual number of environments from the initial observations
        num_envs = setup.initial_obs.shape[0] if setup.initial_obs.ndim > 0 else 1

        return TrainCarry(
            rng=setup.rng,
            algorithm_state=setup.algorithm_state,
            buffer_state=setup.buffer_state,
            env_state=setup.initial_env_state,
            obs=setup.initial_obs,
            env_steps=jnp.zeros(num_envs, dtype=jnp.int32),
            episode_rewards=jnp.zeros(num_envs, dtype=DTYPE),
            total_updates_done=jnp.array(0, dtype=jnp.int32),
        )


@chex.dataclass
class UpdateCarry:
    """Inner carry for per-step parameter updates."""

    rng: chex.PRNGKey
    algorithm_state: AlgorithmState
    buffer_state: BufferStateType
    total_updates_done: chex.Array  # () int32
    chunk_updates_done: chex.Array  # () int32
    actor_loss_ema: chex.Array  # () float32
    critic_loss_ema: chex.Array  # () float32
    alpha_ema: chex.Array  # () float32
    q_ema: chex.Array  # () float32

    @staticmethod
    def init(carry: ChunkCarry, rng: chex.PRNGKey, buffer_state: BufferStateType) -> UpdateCarry:
        return UpdateCarry(
            rng=rng,
            algorithm_state=carry.train.algorithm_state,
            buffer_state=buffer_state,
            total_updates_done=carry.train.total_updates_done,
            chunk_updates_done=carry.chunk_updates_done,
            actor_loss_ema=carry.actor_loss_ema,
            critic_loss_ema=carry.critic_loss_ema,
            alpha_ema=carry.alpha_ema,
            q_ema=carry.q_ema,
        )


@chex.dataclass
class ChunkCarry:
    """Carry through the scan over env steps inside a chunk."""

    train: TrainCarry
    # meters/EMAs are stored here to avoid tuples
    chunk_updates_done: chex.Array  # () int32
    actor_loss_ema: chex.Array  # () float32
    critic_loss_ema: chex.Array  # () float32
    alpha_ema: chex.Array  # () float32
    q_ema: chex.Array  # () float32
    reward_ema: chex.Array  # () float32

    @staticmethod
    def init(train_carry: TrainCarry) -> ChunkCarry:
        from config import DTYPE

        return ChunkCarry(
            train=train_carry,
            chunk_updates_done=jnp.array(0, jnp.int32),
            actor_loss_ema=jnp.array(0.0, DTYPE),
            critic_loss_ema=jnp.array(0.0, DTYPE),
            alpha_ema=jnp.array(0.0, DTYPE),
            q_ema=jnp.array(0.0, DTYPE),
            reward_ema=jnp.array(0.0, DTYPE),
        )


@chex.dataclass
class ChunkSummary:
    """Small scalar summary returned to the host after each chunk."""

    chunk_updates: chex.Array  # () int32
    actor_loss: chex.Array  # () float32
    critic_loss: chex.Array  # () float32
    alpha: chex.Array  # () float32
    q_values: chex.Array  # () float32
    reward: chex.Array  # () float32

    @staticmethod
    def from_carry(carry: ChunkCarry) -> ChunkSummary:
        return ChunkSummary(
            chunk_updates=carry.chunk_updates_done,
            actor_loss=carry.actor_loss_ema,
            critic_loss=carry.critic_loss_ema,
            alpha=carry.alpha_ema,
            q_values=carry.q_ema,
            reward=carry.reward_ema,
        )
