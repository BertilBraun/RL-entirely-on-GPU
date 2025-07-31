"""
Data structures for JAX-based SAC training.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from algorithms.replay_buffer import ReplayBufferState
from algorithms.sac import SAC, SACState
from config import NUM_ENVS
from typing import Union
from environment.cartpole import CartPoleEnv, CartPoleState
from environment.double_pendulum_cartpole import DoublePendulumCartPoleEnv, DoublePendulumCartPoleState

# ----------------------------
# Setup data structures
# ----------------------------


EnvType = DoublePendulumCartPoleEnv | CartPoleEnv


@chex.dataclass
class TrainingSetup:
    """Contains all components needed to initialize training."""

    env: EnvType
    sac: SAC
    sac_state: SACState
    buffer_state: ReplayBufferState
    initial_obs: chex.Array
    initial_env_state: Union[CartPoleState, DoublePendulumCartPoleState]
    rng: chex.PRNGKey


# ----------------------------
# Training data structures
# ----------------------------


@chex.dataclass
class TrainCarry:
    """State that persists across chunks (host <-> device boundary)."""

    rng: chex.PRNGKey
    sac_state: SACState
    buffer_state: ReplayBufferState
    env_state: Union[CartPoleState, DoublePendulumCartPoleState]
    obs: chex.Array  # (num_envs, obs_dim)
    env_steps: chex.Array  # (num_envs,) int32
    episode_rewards: chex.Array  # (num_envs,) float32
    total_updates_done: chex.Array  # () int32

    @staticmethod
    def init(setup: TrainingSetup) -> TrainCarry:
        return TrainCarry(
            rng=setup.rng,
            sac_state=setup.sac_state,
            buffer_state=setup.buffer_state,
            env_state=setup.initial_env_state,
            obs=setup.initial_obs,
            env_steps=jnp.zeros(NUM_ENVS, dtype=jnp.int32),
            episode_rewards=jnp.zeros(NUM_ENVS, dtype=jnp.float32),
            total_updates_done=jnp.array(0, dtype=jnp.int32),
        )


@chex.dataclass
class UpdateCarry:
    """Inner carry for per-step parameter updates."""

    rng: chex.PRNGKey
    sac_state: SACState
    buffer_state: ReplayBufferState
    total_updates_done: chex.Array  # () int32
    chunk_updates_done: chex.Array  # () int32
    actor_loss_ema: chex.Array  # () float32
    critic_loss_ema: chex.Array  # () float32
    alpha_ema: chex.Array  # () float32
    q_ema: chex.Array  # () float32

    @staticmethod
    def init(carry: ChunkCarry, rng: chex.PRNGKey, buffer_state: ReplayBufferState) -> UpdateCarry:
        return UpdateCarry(
            rng=rng,
            sac_state=carry.train.sac_state,
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
        return ChunkCarry(
            train=train_carry,
            chunk_updates_done=jnp.array(0, jnp.int32),
            actor_loss_ema=jnp.array(0.0, jnp.float32),
            critic_loss_ema=jnp.array(0.0, jnp.float32),
            alpha_ema=jnp.array(0.0, jnp.float32),
            q_ema=jnp.array(0.0, jnp.float32),
            reward_ema=jnp.array(0.0, jnp.float32),
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
