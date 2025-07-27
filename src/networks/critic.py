"""
Critic network implementation using Flax.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import chex


class CriticNetwork(nn.Module):
    """
    Critic network that estimates Q-values from state-action pairs.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)

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
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        # Output layer (single Q-value)
        q_value = nn.Dense(1)(x)

        return q_value


class DoubleCriticNetwork(nn.Module):
    """
    Double critic network (Q1 and Q2) for SAC.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)

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
        q1 = self.critic1(obs, action, training=training)
        q2 = self.critic2(obs, action, training=training)

        return q1, q2

    def critic1_forward(self, obs: chex.Array, action: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass of only the first critic."""
        return self.critic1(obs, action, training=training)

    def critic2_forward(self, obs: chex.Array, action: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass of only the second critic."""
        return self.critic2(obs, action, training=training)


def create_critic_network(
    obs_dim: int = 3, action_dim: int = 1, hidden_dims: Tuple[int, ...] = (256, 256), double: bool = True
) -> nn.Module:
    """
    Factory function to create critic network.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        double: Whether to use double critic

    Returns:
        Critic network instance
    """
    if double:
        return DoubleCriticNetwork(hidden_dims=hidden_dims)
    else:
        return CriticNetwork(hidden_dims=hidden_dims)
