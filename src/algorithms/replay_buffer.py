"""
JAX-based replay buffer implementation.
"""

import jax
import chex
import jax.numpy as jnp
from typing import NamedTuple


class Transition(NamedTuple):
    """Single transition for replay buffer."""

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array


class ReplayBufferState(NamedTuple):
    """State of the replay buffer."""

    data: Transition
    size: chex.Array  # Current number of elements
    ptr: chex.Array  # Current write pointer


class ReplayBuffer:
    """
    JAX-based replay buffer with circular buffer logic.
    All operations are JIT-compatible.
    """

    def __init__(self, capacity: int, obs_dim: int = 3, action_dim: int = 1):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def init_buffer_state(self, key: chex.PRNGKey) -> ReplayBufferState:
        """Initialize empty buffer state."""
        # Initialize with zeros
        data = Transition(
            obs=jnp.zeros((self.capacity, self.obs_dim)),
            action=jnp.zeros((self.capacity, self.action_dim)),
            reward=jnp.zeros((self.capacity, 1)),
            next_obs=jnp.zeros((self.capacity, self.obs_dim)),
            done=jnp.zeros((self.capacity, 1), dtype=bool),
        )

        return ReplayBufferState(data=data, size=jnp.array(0), ptr=jnp.array(0))

    @staticmethod
    @jax.jit
    def add(buffer_state: ReplayBufferState, transition: Transition) -> ReplayBufferState:
        """
        Add a transition to the buffer.

        Args:
            buffer_state: Current buffer state
            transition: Transition to add

        Returns:
            Updated buffer state
        """
        # Update data at current pointer
        new_data = Transition(
            obs=buffer_state.data.obs.at[buffer_state.ptr].set(transition.obs.reshape(-1)),
            action=buffer_state.data.action.at[buffer_state.ptr].set(transition.action),
            reward=buffer_state.data.reward.at[buffer_state.ptr].set(transition.reward.reshape(1)),
            next_obs=buffer_state.data.next_obs.at[buffer_state.ptr].set(transition.next_obs.reshape(-1)),
            done=buffer_state.data.done.at[buffer_state.ptr].set(transition.done.reshape(1)),
        )

        # Update pointer and size
        new_ptr = (buffer_state.ptr + 1) % buffer_state.data.obs.shape[0]
        new_size = jnp.minimum(buffer_state.size + 1, buffer_state.data.obs.shape[0])

        return ReplayBufferState(data=new_data, size=new_size, ptr=new_ptr)

    @staticmethod
    @jax.jit
    def add_batch(buffer_state: ReplayBufferState, transitions: Transition) -> ReplayBufferState:
        """
        Add a batch of transitions to the buffer.

        Args:
            buffer_state: Current buffer state
            transitions: Batch of transitions to add

        Returns:
            Updated buffer state
        """
        batch_size = transitions.obs.shape[0]
        capacity = buffer_state.data.obs.shape[0]

        # Calculate indices where data will be written
        start_ptr = buffer_state.ptr
        end_ptr = start_ptr + batch_size

        # Handle wrap-around case
        if end_ptr <= capacity:
            # No wrap-around
            new_data = Transition(
                obs=buffer_state.data.obs.at[start_ptr:end_ptr].set(transitions.obs),
                action=buffer_state.data.action.at[start_ptr:end_ptr].set(transitions.action),
                reward=buffer_state.data.reward.at[start_ptr:end_ptr].set(transitions.reward),
                next_obs=buffer_state.data.next_obs.at[start_ptr:end_ptr].set(transitions.next_obs),
                done=buffer_state.data.done.at[start_ptr:end_ptr].set(transitions.done),
            )
        else:
            # Wrap-around case
            first_part_size = capacity - start_ptr
            second_part_size = batch_size - first_part_size

            new_data = Transition(
                obs=buffer_state.data.obs.at[start_ptr:]
                .set(transitions.obs[:first_part_size])
                .at[:second_part_size]
                .set(transitions.obs[first_part_size:]),
                action=buffer_state.data.action.at[start_ptr:]
                .set(transitions.action[:first_part_size])
                .at[:second_part_size]
                .set(transitions.action[first_part_size:]),
                reward=buffer_state.data.reward.at[start_ptr:]
                .set(transitions.reward[:first_part_size])
                .at[:second_part_size]
                .set(transitions.reward[first_part_size:]),
                next_obs=buffer_state.data.next_obs.at[start_ptr:]
                .set(transitions.next_obs[:first_part_size])
                .at[:second_part_size]
                .set(transitions.next_obs[first_part_size:]),
                done=buffer_state.data.done.at[start_ptr:]
                .set(transitions.done[:first_part_size])
                .at[:second_part_size]
                .set(transitions.done[first_part_size:]),
            )

        # Update pointer and size
        new_ptr = end_ptr % capacity
        new_size = jnp.minimum(buffer_state.size + batch_size, capacity)

        return ReplayBufferState(data=new_data, size=new_size, ptr=new_ptr)

    @staticmethod
    def sample(buffer_state: ReplayBufferState, key: chex.PRNGKey, batch_size: int) -> Transition:
        """
        Sample a batch of transitions from the buffer.

        Args:
            buffer_state: Current buffer state
            key: Random key for sampling
            batch_size: Number of transitions to sample

        Returns:
            Batch of sampled transitions
        """
        # Sample random indices
        indices = jax.random.randint(key, (batch_size,), 0, buffer_state.size)

        # Extract sampled data
        return Transition(
            obs=buffer_state.data.obs[indices],
            action=buffer_state.data.action[indices],
            reward=buffer_state.data.reward[indices],
            next_obs=buffer_state.data.next_obs[indices],
            done=buffer_state.data.done[indices],
        )

    @staticmethod
    def can_sample(buffer_state: ReplayBufferState, batch_size: int) -> bool:
        """Check if buffer has enough data to sample."""
        return buffer_state.size >= batch_size
