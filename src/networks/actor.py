"""
Actor network implementation using Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import chex


class ActorNetwork(nn.Module):
    """
    Actor network that outputs Gaussian policy parameters.
    Uses tanh + reparameterization trick for continuous control.
    """

    hidden_dims: Tuple[int, ...]
    action_dim: int
    max_action: float
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: chex.Array, training: bool = True) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass of actor network.

        Args:
            obs: Observation tensor (..., obs_dim)
            training: Whether in training mode

        Returns:
            Tuple of (mu, log_std) for Gaussian policy
        """
        x = obs

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        # Output layer for mean and log standard deviation
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)

        # Clip log_std to reasonable range
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample_action(
        self, params: chex.ArrayTree, obs: chex.Array, key: chex.PRNGKey, training: bool = True
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Sample action from Gaussian policy with reparameterization trick.

        Args:
            params: Network parameters
            obs: Observation
            key: Random key for sampling
            training: Whether in training mode

        Returns:
            Tuple of (action, log_prob)
        """
        mu, log_std = self.apply(params, obs, training=training)
        std = jnp.exp(log_std)

        # Reparameterization trick
        eps = jax.random.normal(key, mu.shape)
        raw_action = mu + eps * std

        # Apply tanh to bound action
        action = jnp.tanh(raw_action) * self.max_action

        # Compute log probability with tanh correction
        log_prob = self._compute_log_prob(raw_action, mu, log_std)

        return action, log_prob

    def _compute_log_prob(self, raw_action: chex.Array, mu: chex.Array, log_std: chex.Array) -> chex.Array:
        """
        Compute log probability of action under Gaussian policy with tanh squashing.

        Args:
            raw_action: Unsquashed action
            mu: Mean of Gaussian
            log_std: Log standard deviation of Gaussian

        Returns:
            Log probability of the action
        """
        # Gaussian log-prob in raw (unsquashed) space
        inv_std = jnp.exp(-log_std)
        log_prob = -0.5 * (((raw_action - mu) * inv_std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))

        # Numerically stable tanh correction:
        # log(1 - tanh(u)^2) = log 4 - softplus(2u) - softplus(-2u)
        u = raw_action
        log_det_tanh = jnp.log(4.0) - nn.softplus(2.0 * u) - nn.softplus(-2.0 * u)
        log_prob = log_prob - log_det_tanh

        # Sum over action dims
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
        return log_prob

    def deterministic_action(self, params: chex.ArrayTree, obs: chex.Array, training: bool = False) -> chex.Array:
        """
        Get deterministic action (mean of policy).

        Args:
            params: Network parameters
            obs: Observation
            training: Whether in training mode

        Returns:
            Deterministic action
        """
        mu, _ = self.apply(params, obs, training=training)
        return jnp.tanh(mu) * self.max_action
