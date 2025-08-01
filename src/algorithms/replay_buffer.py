"""
JAX-based replay buffer implementation.
"""

import jax
import chex
import jax.numpy as jnp


@chex.dataclass
class Transition:
    """Single transition for replay buffer."""

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array


@chex.dataclass
class ReplayBufferState:
    """State of the replay buffer."""

    data: Transition
    size: chex.Array  # Current number of elements
    ptr: chex.Array  # Current write pointer


class ReplayBuffer:
    """
    JAX-based replay buffer with circular buffer logic.
    All operations are JIT-compatible.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def init_buffer_state(self) -> ReplayBufferState:
        """Initialize empty buffer state."""
        from config import DTYPE

        # Initialize with zeros
        data = Transition(
            obs=jnp.zeros((self.capacity, self.obs_dim), dtype=DTYPE),
            action=jnp.zeros((self.capacity, self.action_dim), dtype=DTYPE),
            reward=jnp.zeros((self.capacity, 1), dtype=DTYPE),
            next_obs=jnp.zeros((self.capacity, self.obs_dim), dtype=DTYPE),
            done=jnp.zeros((self.capacity, 1), dtype=bool),
        )

        return ReplayBufferState(data=data, size=jnp.array(0, dtype=jnp.int32), ptr=jnp.array(0, dtype=jnp.int32))

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
        batch_size = transitions.obs.shape[0]  # type: ignore
        capacity = buffer_state.data.obs.shape[0]  # type: ignore

        # Indices have a compile-time-known length (batch_size), then we shift by the runtime pointer.
        idx = (jnp.arange(batch_size, dtype=jnp.int32) + buffer_state.ptr) % capacity

        def set_field(buf_arr, new_arr):
            return buf_arr.at[idx].set(new_arr.reshape(batch_size, -1))

        new_data = Transition(
            obs=set_field(buffer_state.data.obs, transitions.obs),
            action=set_field(buffer_state.data.action, transitions.action),
            reward=set_field(buffer_state.data.reward, transitions.reward),
            next_obs=set_field(buffer_state.data.next_obs, transitions.next_obs),
            done=set_field(buffer_state.data.done, transitions.done),
        )

        new_ptr = (buffer_state.ptr + batch_size) % capacity
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
        # Ensure randint(high) is valid; when size==0 we sample 0.
        size = jnp.maximum(buffer_state.size, 1)
        indices = jax.random.randint(key, (batch_size,), 0, size)

        # Extract sampled data
        return Transition(
            obs=buffer_state.data.obs[indices],  # type: ignore
            action=buffer_state.data.action[indices],  # type: ignore
            reward=buffer_state.data.reward[indices],  # type: ignore
            next_obs=buffer_state.data.next_obs[indices],  # type: ignore
            done=buffer_state.data.done[indices],  # type: ignore
        )

    @staticmethod
    def can_sample(buffer_state: ReplayBufferState, batch_size: int) -> bool:
        """Check if buffer has enough data to sample."""
        return buffer_state.size >= batch_size  # type: ignore
