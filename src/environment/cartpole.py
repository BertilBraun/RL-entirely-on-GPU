from functools import partial
import jax
import chex
import jax.numpy as jnp
from typing import Tuple


# Default physical parameters
DEFAULT_PARAMS = {
    'dt': 1 / 60,
    'g': 9.81,
    'length': 1.0,
    'm': 0.01,  # pendulum mass
    'M': 1.0,  # cart mass
    'max_speed': 4.0,
    'max_base_speed': 3.0,
    'max_force': 15.0,
    'rail_limit': 4.0,  # base can move between -4 and 4
    'theta_damp': 0.00,  # pole rotational damping
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
    state: CartPoleState,
    force: chex.Array,
    dt: float,
    g: float,
    length: float,
    m: float,
    M: float,
    rail_limit: float,
    theta_damp: float,
) -> CartPoleState:
    """
    Traditional cart-pole dynamics with simple boundary handling.
    """
    cos_t = jnp.cos(state.theta)
    sin_t = jnp.sin(state.theta)
    total_mass = M + m

    # Standard cart-pole dynamics
    temp = (force + m * length * state.theta_dot**2 * sin_t) / total_mass
    num = g * sin_t - cos_t * temp
    den = length * (4.0 / 3.0 - m * cos_t**2 / total_mass)
    theta_ddot = num / den
    x_ddot = temp - (m * length * theta_ddot * cos_t) / total_mass

    # Apply damping
    theta_ddot = theta_ddot - theta_damp * state.theta_dot

    # Semi-implicit Euler integration
    x_dot_new = state.x_dot + dt * x_ddot
    x_new = state.x + dt * x_dot_new
    theta_dot_new = state.theta_dot + dt * theta_ddot
    theta_new = state.theta + dt * theta_dot_new

    # Wrap angle to [-pi, pi]
    theta_new = jnp.mod(theta_new + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    return CartPoleState(x=x_new, x_dot=x_dot_new, theta=theta_new, theta_dot=theta_dot_new)


@jax.jit
def get_obs(state: CartPoleState, rail_limit: float, max_base_speed: float, max_speed: float) -> chex.Array:
    """Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]"""
    return jnp.concatenate(
        [
            state.x / rail_limit,  # Normalize position to [-1, 1]
            state.x_dot / max_base_speed,  # Normalize base velocity to [-1, 1]
            jnp.cos(state.theta),  # Cosine of angle
            jnp.sin(state.theta),  # Sine of angle
            state.theta_dot / max_speed,  # Normalize angular velocity to [-1, 1]
        ],
        axis=-1,
    )


@jax.jit
def reward_fn(state: CartPoleState, force: chex.Array, length: float, rail_limit: float) -> chex.Array:
    """
    Smooth reward favoring upright pole, small velocities, and gentle control.
    """
    upright = 3.0 * jnp.exp(-(state.theta**2) / (0.12**2))  # Large reward for pole being upright

    r = (
        jnp.cos(state.theta)
        + upright  # Large reward for pole being upright
        - 0.05 * (state.x**2)
        - 0.01 * (state.theta_dot**2)
        - 0.01 * (state.x_dot**2)
        - 1e-4 * (force**2)
        - jnp.where(jnp.abs(state.x) >= rail_limit - 0.5, 2, 0)  # penalty for hitting the boundary
    )
    return r.squeeze(-1)


@jax.jit
def is_done(state: CartPoleState, rail_limit: float) -> chex.Array:
    """Episode is done when cart hits boundary."""
    return (jnp.abs(state.x) >= rail_limit).squeeze(-1)


class CartPoleEnv:
    """
    JAX-based vectorized cart-pole environment with traditional physics.
    Simple boundary handling - episode ends when cart hits rails.
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
        theta_damp: float = DEFAULT_PARAMS['theta_damp'],
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
        self.theta_damp = theta_damp

        # Action and observation spaces
        self.action_dim = 1  # Force applied to base
        self.obs_dim = 5  # [x, x_dot, cos(theta), sin(theta), theta_dot]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, reset_key: chex.PRNGKey) -> Tuple[chex.Array, CartPoleState]:
        """Reset environment(s) to initial state."""

        def rand(shape, lo, hi, key):
            key, sub_key = jax.random.split(key)
            return jax.random.uniform(sub_key, shape, minval=lo, maxval=hi)

        key_x, key_x_dot, key_theta, key_theta_neg, key_theta_dot = jax.random.split(reset_key, 5)

        x = rand((self.num_envs, 1), -1.0, 1.0, key_x)
        x_dot = rand((self.num_envs, 1), -0.5, 0.5, key_x_dot)
        theta = rand((self.num_envs, 1), jnp.pi / 2, jnp.pi, key_theta)
        negative_theta = rand((self.num_envs, 1), 0, 1, key_theta_neg) > 0.5
        theta = jnp.where(negative_theta, -theta, theta)
        theta_dot = rand((self.num_envs, 1), -0.5, 0.5, key_theta_dot)

        state = CartPoleState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        obs = get_obs(state, self.rail_limit, self.max_base_speed, self.max_speed)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: CartPoleState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, CartPoleState]:
        """Step environment(s) forward."""
        # Clip action to valid range
        action = jnp.asarray(action).reshape(self.num_envs, 1)
        force = jnp.clip(action, -self.max_force, self.max_force)

        # Physics step
        next_state = cartpole_step(
            state=state,
            force=force,
            dt=self.dt,
            g=self.g,
            length=self.length,
            m=self.m,
            M=self.M,
            rail_limit=self.rail_limit,
            theta_damp=self.theta_damp,
        )

        # Clip velocities (safety caps)
        next_state.x_dot = jnp.clip(next_state.x_dot, -self.max_base_speed, self.max_base_speed)
        next_state.theta_dot = jnp.clip(next_state.theta_dot, -self.max_speed, self.max_speed)

        # Check if episode is done (hit boundary)
        done = is_done(next_state, self.rail_limit)

        # Reward & outputs
        reward = reward_fn(state, force, self.length, self.rail_limit)
        next_obs = get_obs(next_state, self.rail_limit, self.max_base_speed, self.max_speed)

        return next_obs, reward, done, next_state
