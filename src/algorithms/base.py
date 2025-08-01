"""
Abstract base algorithm interface for RL algorithms in JAX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Tuple, Protocol
import chex


class AlgorithmState(Protocol):
    """Protocol for algorithm state that supports saving/loading."""

    def save_model(self, step: int) -> None:
        """Save model parameters."""
        ...

    def try_load(self) -> AlgorithmState:
        """Try to load model parameters."""
        ...


class AlgorithmInfo(Protocol):
    """Protocol for algorithm training info/metrics."""

    pass


class Algorithm(ABC):
    """
    Abstract base class for RL algorithms.

    Defines the common interface that all algorithms (SAC, PPO, etc.) must implement
    to be compatible with the training infrastructure.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_action: float) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

    @abstractmethod
    def init_state(self, key: chex.PRNGKey) -> AlgorithmState:
        """Initialize algorithm state with network parameters and optimizers."""
        pass

    @abstractmethod
    def update_step(self, state: Any, batch: Any, key: chex.PRNGKey) -> Any:
        """
        Perform a single training update step.

        Args:
            state: Current algorithm state
            batch: Training batch (format depends on algorithm)
            key: Random key for stochastic operations

        Returns:
            Tuple of (new_state, training_info)
        """
        pass

    @abstractmethod
    def select_action_stochastic(self, state: Any, obs: chex.Array, key: chex.PRNGKey) -> Any:
        """Select stochastic action given observation."""
        pass

    @abstractmethod
    def select_action_deterministic(self, state: Any, obs: chex.Array) -> Any:
        """Select deterministic action given observation."""
        pass

    @abstractmethod
    def collect_rollout_data(self, state: Any, obs: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, Any]:
        """
        Collect data for rollout (algorithm-specific).

        Args:
            state: Algorithm state
            obs: Observations
            key: Random key

        Returns:
            Tuple of (action, rollout_data) where rollout_data contains
            algorithm-specific information needed for buffer storage
        """
        pass

    @abstractmethod
    def update_buffer(
        self,
        buffer_state: Any,
        rollout_data: Any,
        obs: chex.Array,
        action: chex.Array,
        reward: chex.Array,
        next_obs: chex.Array,
        done: chex.Array,
    ) -> Any:
        """
        Update the algorithm's buffer with new transition data.

        Args:
            buffer_state: Current buffer state
            rollout_data: Algorithm-specific data from collect_rollout_data
            obs, action, reward, next_obs, done: Standard transition data

        Returns:
            Updated buffer state
        """
        pass

    @abstractmethod
    def prepare_update_batch(self, buffer_state: Any, key: chex.PRNGKey, batch_size: int) -> Any:
        """
        Prepare a batch for training from the buffer.

        Args:
            buffer_state: Current buffer state
            key: Random key for sampling
            batch_size: Requested batch size

        Returns:
            Training batch in algorithm-specific format
        """
        pass

    def get_value(self, state: Any, obs: chex.Array) -> chex.Array:
        """
        Get value estimate for observations (for PPO).
        Default implementation raises NotImplementedError - only PPO needs this.
        """
        raise NotImplementedError(f'{self.algorithm_name} does not implement value estimation')

    def get_action_and_value(
        self, state: Any, obs: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Get action, log prob, and value for observations (used during PPO rollout).
        Default implementation raises NotImplementedError - only PPO needs this.
        """
        raise NotImplementedError(f'{self.algorithm_name} does not implement get_action_and_value')

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        pass

    @property
    @abstractmethod
    def requires_replay_buffer(self) -> bool:
        """Return True if algorithm uses replay buffer (off-policy), False if episode buffer (on-policy)."""
        pass


class AlgorithmConfig(NamedTuple):
    """Base configuration class for algorithms."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    grad_clip: float = 10.0
