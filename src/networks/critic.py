"""
Critic network implementation using Flax.
"""

import chex
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from config import DTYPE


class CriticNetwork(nn.Module):
    """
    Critic network that estimates Q-values from state-action pairs.
    """

    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: chex.Array, action: chex.Array, training: bool = True) -> chex.Array:
        """
        Forward pass of critic network.

        Args:
            obs: Observation tensor (..., obs_dim)
            action: Action tensor (..., action_dim)
            training: Whether in training mode

        Returns:
            Q-value scalar (..., 1)
        """
        # Concatenate observation and action
        x = jnp.concatenate([obs, action], axis=-1)

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=DTYPE)(x)
            x = nn.relu(x)

        # Output layer (single Q-value)
        q_value = nn.Dense(1, dtype=DTYPE)(x)

        return q_value


class DoubleCriticNetwork(nn.Module):
    """
    Double critic network (Q1 and Q2) for SAC.
    """

    hidden_dims: Tuple[int, ...]

    def setup(self):
        """Initialize two critic networks."""
        self.critic1 = CriticNetwork(hidden_dims=self.hidden_dims)
        self.critic2 = CriticNetwork(hidden_dims=self.hidden_dims)

    def __call__(self, obs: chex.Array, action: chex.Array, training: bool = True) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of both critic networks.

        Args:
            obs: Observation tensor
            action: Action tensor
            training: Whether in training mode

        Returns:
            Tuple of (q1_value, q2_value)
        """
        q1 = self.critic1(obs.astype(DTYPE), action.astype(DTYPE), training=training)
        q2 = self.critic2(obs.astype(DTYPE), action.astype(DTYPE), training=training)

        return q1, q2
