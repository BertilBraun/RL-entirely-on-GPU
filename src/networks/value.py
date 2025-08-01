"""
Value network implementation for PPO using Flax.
"""

import chex
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from config import DTYPE


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values V(s) from observations.
    Used in PPO for advantage estimation and value function updates.
    """

    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: chex.Array, training: bool = True) -> chex.Array:
        """
        Forward pass of value network.

        Args:
            obs: Observation tensor (..., obs_dim)
            training: Whether in training mode

        Returns:
            State value scalar (..., 1)
        """
        x = obs.astype(DTYPE)

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=DTYPE)(x)
            x = nn.relu(x)

        # Output layer (single value)
        value = nn.Dense(1, dtype=DTYPE)(x)

        return value
