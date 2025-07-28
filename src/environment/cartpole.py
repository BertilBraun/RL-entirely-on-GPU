"""
JAX-based vectorized cart-pole environment implementation.
"""

import jax
import chex
import jax.numpy as jnp
from typing import Tuple


# Default physical parameters - centralized in one place
DEFAULT_PARAMS = {
    'dt': 0.05,
    'g': 9.81,
    'length': 1.0,
    'm': 1.0,  # pendulum mass
    'M': 1.0,  # base/cart mass
    'damping': 0.99,
    'max_speed': 8.0,
    'max_base_speed': 5.0,
    'max_force': 10.0,
    'rail_limit': 2.0,  # base can move between -2 and 2
}


@chex.dataclass
class CartPoleState:
    """State representation for cart-pole environment."""

    x: chex.Array  # Base position
    x_dot: chex.Array  # Base velocity
    theta: chex.Array  # Angular position (from vertical)
    theta_dot: chex.Array  # Angular velocity


@jax.jit
def cartpole_step(
    x: chex.Array,
    x_dot: chex.Array,
    theta: chex.Array,
    theta_dot: chex.Array,
    force: chex.Array,
    dt: float,
    g: float,
    length: float,
    m: float,
    M: float,
    damping: float,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Vectorized cart-pole physics step using JAX with damping.

    Args:
        x: Base position(s)
        x_dot: Base velocity(ies)
        theta: Angular position(s) from vertical in radians
        theta_dot: Angular velocity(ies) in rad/s
        force: Applied force(s) to the base
        dt: Time step
        g: Gravitational acceleration
        length: Pendulum length
        m: Pendulum mass
        M: Base/cart mass
        damping: Damping coefficient

    Returns:
        Tuple of (next_x, next_x_dot, next_theta, next_theta_dot)
    """
    # Cart-pole dynamics
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Total mass
    total_mass = M + m

    # Temporary variable for common term
    temp = (force + m * length * theta_dot**2 * sin_theta) / total_mass

    # Angular acceleration
    numerator = g * sin_theta - cos_theta * temp
    denominator = length * (4.0 / 3.0 - m * cos_theta**2 / total_mass)
    theta_ddot = numerator / denominator

    # Linear acceleration
    x_ddot = temp - m * length * theta_ddot * cos_theta / total_mass

    # Forward Euler integration with damping
    x_dot_new = x_dot + dt * x_ddot * damping
    x_new = x + dt * x_dot_new

    theta_dot_new = theta_dot + dt * theta_ddot * damping
    theta_new = theta + dt * theta_dot_new

    # Constrain base position to rail limits
    x_new = jnp.clip(x_new, -DEFAULT_PARAMS['rail_limit'], DEFAULT_PARAMS['rail_limit'])
    # if at the limit, set x_dot to 0
    x_dot_new = jnp.where(jnp.abs(x_new) >= DEFAULT_PARAMS['rail_limit'], 0.0, x_dot_new)

    # Wrap angle to [-π, π]
    theta_new = jnp.mod(theta_new + jnp.pi, 2 * jnp.pi) - jnp.pi

    return x_new, x_dot_new, theta_new, theta_dot_new


@jax.jit
def get_obs(x: chex.Array, x_dot: chex.Array, theta: chex.Array, theta_dot: chex.Array) -> chex.Array:
    """
    Convert cart-pole state to observation.

    Args:
        x: Base position(s)
        x_dot: Base velocity(ies)
        theta: Angular position(s)
        theta_dot: Angular velocity(ies)

    Returns:
        Observation array [x, x_dot, cos(theta), sin(theta), theta_dot]
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    return jnp.concatenate([x, x_dot, cos_theta, sin_theta, theta_dot], axis=-1)


@jax.jit
def reward_fn(
    x: chex.Array,
    x_dot: chex.Array,
    theta: chex.Array,
    theta_dot: chex.Array,
    force: chex.Array,
    length: float,
) -> chex.Array:
    """
    Compute reward for cart-pole state and action.
    Reward is 1.0 when pendulum is in top 10% reachable height, 0.0 otherwise.

    Args:
        x: Base position(s)
        x_dot: Base velocity(ies)
        theta: Angular position(s) from vertical
        theta_dot: Angular velocity(ies)
        force: Applied force(s)
        length: Pendulum length

    Returns:
        Reward value(s)
    """
    # Calculate pendulum tip height (y-coordinate)
    # y = l * cos(theta) when theta=0 is vertical upward
    y_tip = length * jnp.cos(theta)

    return y_tip

    # Reward threshold: top 10% means y > 0.9 * l
    reward_threshold = 0.9 * length

    # Binary reward: 1.0 if in top 10%, 0.0 otherwise
    reward = jnp.where(y_tip > reward_threshold, 1.0, 0.0)

    return reward


class CartPoleEnv:
    """
    JAX-based vectorized cart-pole environment.
    Base moves on rail between -2 and 2, pendulum hangs from base.
    """

    def __init__(
        self,
        num_envs: int,
        max_base_speed: float = DEFAULT_PARAMS['max_base_speed'],
        max_speed: float = DEFAULT_PARAMS['max_speed'],
        max_force: float = DEFAULT_PARAMS['max_force'],
        rail_limit: float = DEFAULT_PARAMS['rail_limit'],
        dt: float = DEFAULT_PARAMS['dt'],
        g: float = DEFAULT_PARAMS['g'],
        length: float = DEFAULT_PARAMS['length'],
        m: float = DEFAULT_PARAMS['m'],
        M: float = DEFAULT_PARAMS['M'],
        damping: float = DEFAULT_PARAMS['damping'],
    ):
        assert num_envs > 0, 'num_envs must be at least 1'

        self.num_envs = num_envs
        self.max_base_speed = max_base_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.rail_limit = rail_limit
        self.dt = dt
        self.g = g
        self.length = length
        self.m = m
        self.M = M
        self.damping = damping

        # Action and observation spaces
        self.action_dim = 1  # Force applied to base
        self.obs_dim = 5  # [x, x_dot, cos(theta), sin(theta), theta_dot]

        # Initialize random key
        self.key = jax.random.PRNGKey(42)

    def reset(self) -> Tuple[chex.Array, CartPoleState]:
        """Reset environment(s) to initial state."""
        self.key, reset_key = jax.random.split(self.key)

        # Generate keys for each environment
        reset_key, key = jax.random.split(reset_key)
        x = jax.random.uniform(key, (self.num_envs, 1), minval=-1.0, maxval=1.0)
        reset_key, key = jax.random.split(reset_key)
        theta_dot = jax.random.uniform(key, (self.num_envs, 1), minval=-0.5, maxval=0.5)
        reset_key, key = jax.random.split(reset_key)
        theta = jax.random.uniform(key, (self.num_envs, 1), minval=-0.2, maxval=0.2)  # Start near vertical
        reset_key, key = jax.random.split(reset_key)
        x_dot = jax.random.uniform(key, (self.num_envs, 1), minval=-0.5, maxval=0.5)

        state = CartPoleState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        obs = get_obs(x, x_dot, theta, theta_dot)

        return obs, state

    def step(
        self, state: CartPoleState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, CartPoleState]:
        """Step environment(s) forward."""
        # Clip action to valid range
        force = jnp.clip(action, -self.max_force, self.max_force)

        # Physics step
        next_x, next_x_dot, next_theta, next_theta_dot = cartpole_step(
            state.x,
            state.x_dot,
            state.theta,
            state.theta_dot,
            force,
            self.dt,
            self.g,
            self.length,
            self.m,
            self.M,
            self.damping,
        )

        # Clip velocities
        next_x_dot = jnp.clip(next_x_dot, -self.max_base_speed, self.max_base_speed)
        next_theta_dot = jnp.clip(next_theta_dot, -self.max_speed, self.max_speed)

        # Compute reward
        reward = reward_fn(state.x, state.x_dot, state.theta, state.theta_dot, force, self.length)

        # Get next observation
        next_obs = get_obs(next_x, next_x_dot, next_theta, next_theta_dot)

        # Cart-pole environments typically don't terminate
        done = jnp.zeros((self.num_envs,), dtype=bool)

        next_state = CartPoleState(x=next_x, x_dot=next_x_dot, theta=next_theta, theta_dot=next_theta_dot)

        return next_obs, reward, done, next_state
