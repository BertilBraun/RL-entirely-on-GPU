"""
Episode buffer for on-policy PPO training with GAE advantage estimation.
"""

from __future__ import annotations

from typing import NamedTuple
import chex
import jax
import jax.numpy as jnp

# Define DTYPE locally to avoid circular import
DTYPE = jnp.float32


@chex.dataclass
class Episode:
    """Single episode data for PPO training."""

    obs: chex.Array  # (max_steps, num_envs, obs_dim)
    actions: chex.Array  # (max_steps, num_envs, action_dim)
    rewards: chex.Array  # (max_steps, num_envs)
    values: chex.Array  # (max_steps, num_envs)
    log_probs: chex.Array  # (max_steps, num_envs)
    dones: chex.Array  # (max_steps, num_envs) - boolean mask
    length: chex.Array  # (num_envs,) - actual episode length per env


@chex.dataclass
class EpisodeBufferState:
    """State of the episode buffer."""

    episodes: Episode
    current_step: chex.Array  # () int32 - current step in collection
    num_complete_episodes: chex.Array  # () int32 - number of complete episodes


class EpisodeBuffer:
    """
    Episode buffer for PPO that collects full episodes and computes advantages using GAE.
    """

    def __init__(self, max_episode_length: int, obs_dim: int, action_dim: int, num_envs: int) -> None:
        self.max_episode_length = max_episode_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs

    def init_buffer_state(self) -> EpisodeBufferState:
        """Initialize empty episode buffer."""
        episodes = Episode(
            obs=jnp.zeros((self.max_episode_length, self.num_envs, self.obs_dim), dtype=DTYPE),
            actions=jnp.zeros((self.max_episode_length, self.num_envs, self.action_dim), dtype=DTYPE),
            rewards=jnp.zeros((self.max_episode_length, self.num_envs), dtype=DTYPE),
            values=jnp.zeros((self.max_episode_length, self.num_envs), dtype=DTYPE),
            log_probs=jnp.zeros((self.max_episode_length, self.num_envs), dtype=DTYPE),
            dones=jnp.zeros((self.max_episode_length, self.num_envs), dtype=bool),
            length=jnp.zeros(self.num_envs, dtype=jnp.int32),
        )

        return EpisodeBufferState(
            episodes=episodes,
            current_step=jnp.array(0, dtype=jnp.int32),
            num_complete_episodes=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    @jax.jit
    def add_step(
        buffer_state: EpisodeBufferState,
        obs: chex.Array,
        action: chex.Array,
        reward: chex.Array,
        value: chex.Array,
        log_prob: chex.Array,
        done: chex.Array,
    ) -> EpisodeBufferState:
        """Add a single step to all episodes in the buffer."""
        step = buffer_state.current_step

        # Update episodes with new data
        episodes = Episode(
            obs=buffer_state.episodes.obs.at[step].set(obs),
            actions=buffer_state.episodes.actions.at[step].set(action),
            rewards=buffer_state.episodes.rewards.at[step].set(reward),
            values=buffer_state.episodes.values.at[step].set(value),
            log_probs=buffer_state.episodes.log_probs.at[step].set(log_prob),
            dones=buffer_state.episodes.dones.at[step].set(done),
            length=jnp.where(done, step + 1, buffer_state.episodes.length),
        )

        return EpisodeBufferState(
            episodes=episodes,
            current_step=step + 1,
            num_complete_episodes=buffer_state.num_complete_episodes
            + jnp.sum(done.astype(jnp.int32)).astype(jnp.int32),
        )

    @staticmethod
    @jax.jit
    def compute_advantages_and_returns(
        episodes: Episode,
        final_values: chex.Array,  # Values for the final observations
        gamma: float,
        gae_lambda: float,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Args:
            episodes: Episode data
            final_values: Value estimates for final observations (for incomplete episodes)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (advantages, returns) with shape (max_steps, num_envs)
        """
        rewards = episodes.rewards
        values = episodes.values
        dones = episodes.dones.astype(DTYPE)

        # Append final values for bootstrapping
        values_with_final = jnp.concatenate([values, final_values[None, :]], axis=0)

        # Compute deltas (TD errors)
        deltas = rewards + gamma * values_with_final[1:] * (1 - dones) - values_with_final[:-1]

        # Compute advantages using GAE
        advantages = jnp.zeros_like(rewards)
        gae = 0.0

        # Reverse iteration to compute GAE
        def compute_gae_step(carry: chex.Array, inputs: tuple[chex.Array, chex.Array]) -> tuple[chex.Array, chex.Array]:
            gae, _ = carry
            delta, done = inputs
            gae = delta + gamma * gae_lambda * gae * (1 - done)
            return gae, gae

        # Process in reverse order
        _, advantages_reversed = jax.lax.scan(
            compute_gae_step,
            init=0.0,
            xs=(jnp.flip(deltas, axis=0), jnp.flip(dones, axis=0)),
            reverse=False,
        )

        advantages = jnp.flip(advantages_reversed, axis=0)

        # Compute returns
        returns = advantages + values

        return advantages, returns

    @staticmethod
    @jax.jit
    def clear_buffer(buffer_state: EpisodeBufferState) -> EpisodeBufferState:
        """Clear the buffer after processing episodes."""
        episodes = Episode(
            obs=jnp.zeros_like(buffer_state.episodes.obs),
            actions=jnp.zeros_like(buffer_state.episodes.actions),
            rewards=jnp.zeros_like(buffer_state.episodes.rewards),
            values=jnp.zeros_like(buffer_state.episodes.values),
            log_probs=jnp.zeros_like(buffer_state.episodes.log_probs),
            dones=jnp.zeros_like(buffer_state.episodes.dones),
            length=jnp.zeros_like(buffer_state.episodes.length),
        )

        return EpisodeBufferState(
            episodes=episodes,
            current_step=jnp.array(0, dtype=jnp.int32),
            num_complete_episodes=jnp.array(0, dtype=jnp.int32),
        )
