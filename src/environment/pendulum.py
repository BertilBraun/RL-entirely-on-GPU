"""
JAX-based vectorized pendulum environment implementation.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
import chex


# Default physical parameters - centralized in one place
DEFAULT_PARAMS = {
    'dt': 0.05,
    'g': 9.81,
    'l': 1.0,
    'm': 1.0,
    'damping': 0.1,  # Added damping coefficient
    'max_speed': 8.0,
    'max_torque': 2.0,
}


class PendulumState(NamedTuple):
    """State representation for pendulum environment."""

    theta: chex.Array  # Angular position
    theta_dot: chex.Array  # Angular velocity


@jax.jit
def pendulum_step(
    theta: chex.Array,
    theta_dot: chex.Array,
    torque: chex.Array,
    dt: float = DEFAULT_PARAMS['dt'],
    g: float = DEFAULT_PARAMS['g'],
    l: float = DEFAULT_PARAMS['l'],
    m: float = DEFAULT_PARAMS['m'],
    damping: float = DEFAULT_PARAMS['damping'],
) -> Tuple[chex.Array, chex.Array]:
    """
    Vectorized pendulum physics step using JAX with damping.

    Args:
        theta: Angular position(s) in radians
        theta_dot: Angular velocity(ies) in rad/s
        torque: Applied torque(s)
        dt: Time step
        g: Gravitational acceleration
        l: Pendulum length
        m: Pendulum mass
        damping: Damping coefficient

    Returns:
        Tuple of (next_theta, next_theta_dot)
    """
    # Pendulum dynamics with damping: theta_ddot = (g/l) * sin(theta) + torque / (m * l^2) - damping * theta_dot
    theta_ddot = (g / l) * jnp.sin(theta) + torque / (m * l**2) - damping * theta_dot

    # Forward Euler integration
    theta_dot_new = theta_dot + dt * theta_ddot
    theta_new = theta + dt * theta_dot_new

    # Wrap angle to [-π, π]
    theta_new = jnp.mod(theta_new + jnp.pi, 2 * jnp.pi) - jnp.pi

    return theta_new, theta_dot_new


@jax.jit
def get_obs(theta: chex.Array, theta_dot: chex.Array) -> chex.Array:
    """
    Convert pendulum state to observation.

    Args:
        theta: Angular position(s)
        theta_dot: Angular velocity(ies)

    Returns:
        Observation array [cos(theta), sin(theta), theta_dot]
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Stack along last axis to create observation vector
    if theta.ndim == 0:  # Single environment
        return jnp.array([cos_theta, sin_theta, theta_dot])
    else:  # Batched environments
        return jnp.stack([cos_theta, sin_theta, theta_dot], axis=-1)


@jax.jit
def reward_fn(theta: chex.Array, theta_dot: chex.Array, torque: chex.Array) -> chex.Array:
    """
    Compute reward for pendulum state and action.

    Args:
        theta: Angular position(s)
        theta_dot: Angular velocity(ies)
        torque: Applied torque(s)

    Returns:
        Reward value(s)
    """
    # Reward function: r = -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
    angle_cost = theta**2
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.001 * torque**2

    return -(angle_cost + velocity_cost + control_cost)


class PendulumEnv:
    """
    JAX-based vectorized pendulum environment.
    """

    def __init__(
        self,
        num_envs: int = 1,
        max_speed: float = DEFAULT_PARAMS['max_speed'],
        max_torque: float = DEFAULT_PARAMS['max_torque'],
        dt: float = DEFAULT_PARAMS['dt'],
        g: float = DEFAULT_PARAMS['g'],
        l: float = DEFAULT_PARAMS['l'],
        m: float = DEFAULT_PARAMS['m'],
        damping: float = DEFAULT_PARAMS['damping'],
    ):
        self.num_envs = num_envs
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.l = l
        self.m = m
        self.damping = damping

        # Action and observation spaces
        self.action_dim = 1
        self.obs_dim = 3

        # Initialize random key
        self.key = jax.random.PRNGKey(42)

        # Create vectorized functions using vmap
        if num_envs > 1:
            self.reset_fn = jax.jit(jax.vmap(self._reset_single, in_axes=(0,)))
            self.step_fn = jax.jit(jax.vmap(self._step_single, in_axes=(0, 0)))
        else:
            self.reset_fn = jax.jit(self._reset_single)
            self.step_fn = jax.jit(self._step_single)

    @staticmethod
    def _reset_single(key: chex.PRNGKey) -> Tuple[chex.Array, PendulumState]:
        """Reset a single environment."""
        # Random initial angles and velocities
        theta = jax.random.uniform(key, (), minval=-jnp.pi, maxval=jnp.pi)
        key, subkey = jax.random.split(key)
        theta_dot = jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0)

        state = PendulumState(theta=theta, theta_dot=theta_dot)
        obs = get_obs(theta, theta_dot)

        return obs, state

    def _step_single(
        self, state: PendulumState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, PendulumState]:
        """Step a single environment."""
        # Clip action to valid range
        torque = jnp.clip(action, -self.max_torque, self.max_torque)

        # Physics step
        next_theta, next_theta_dot = pendulum_step(
            state.theta, state.theta_dot, torque, self.dt, self.g, self.l, self.m, self.damping
        )

        # Clip velocity
        next_theta_dot = jnp.clip(next_theta_dot, -self.max_speed, self.max_speed)

        # Compute reward
        reward = reward_fn(state.theta, state.theta_dot, torque)

        # Get next observation
        next_obs = get_obs(next_theta, next_theta_dot)

        # Pendulum environments typically don't terminate
        done = jnp.array(False)

        next_state = PendulumState(theta=next_theta, theta_dot=next_theta_dot)

        return next_obs, reward, done, next_state

    def reset(self, key: chex.PRNGKey | None = None) -> Tuple[chex.Array, PendulumState]:
        """Reset environment(s) to initial state."""
        if key is None:
            key = self.key
            self.key, key = jax.random.split(self.key)

        if self.num_envs > 1:
            # Generate keys for each environment
            keys = jax.random.split(key, self.num_envs)
            return self.reset_fn(keys)
        else:
            return self.reset_fn(key)

    def step(
        self, state: PendulumState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, PendulumState]:
        """Step environment(s) forward."""
        if self.num_envs > 1:
            # For multiple environments, action should be (num_envs, action_dim)
            return self.step_fn(state, action)
        else:
            # For single environment, ensure action is scalar
            if action.ndim > 0 and action.shape[0] == 1:
                action = action[0]
            return self.step_fn(state, action)


# Vectorized versions - now using vmap automatically
batched_pendulum_step = jax.vmap(pendulum_step, in_axes=(0, 0, 0, None, None, None, None, None))
batched_get_obs = jax.vmap(get_obs, in_axes=(0, 0))
batched_reward_fn = jax.vmap(reward_fn, in_axes=(0, 0, 0))
