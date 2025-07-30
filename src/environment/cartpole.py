"""
JAX-based vectorized cart-pole with position-dependent viscous damping.
"""

import jax
import chex
import jax.numpy as jnp
from typing import Tuple


# Default physical & damping parameters
DEFAULT_PARAMS = {
    'dt': 1 / 60,
    'g': 9.81,
    'length': 1.0,
    'm': 1.0,  # pendulum mass
    'M': 1.0,  # cart mass
    'max_speed': 8.0,
    'max_base_speed': 5.0,
    'max_force': 10.0,
    'rail_limit': 2.0,  # base can move between -2 and 2
    # --- New viscous damping knobs ---
    # Cart viscous damping = x_damp_base + x_damp_edge_k * (|x|/rail_limit)**x_damp_edge_p
    'x_damp_base': 0.0,
    'x_damp_edge_k': 5.0,
    'x_damp_edge_p': 2.0,
    # Pole viscous damping (small, constant)
    'theta_damp': 0.05,
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
    rail_limit: float,
    x_damp_base: float,
    x_damp_edge_k: float,
    x_damp_edge_p: float,
    theta_damp: float,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Vectorized cart-pole step with position-dependent viscous damping on the cart,
    and small viscous damping on the pole. No bounce at the rails.
    """
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    total_mass = M + m

    # Position-dependent viscous coefficient c(x)
    t = jnp.clip(jnp.abs(x) / rail_limit, 0.0, 1.0)
    c_x = x_damp_base + x_damp_edge_k * (t**x_damp_edge_p)  # [N*s/m] effective

    # Apply damping as a FORCE before solving the coupled dynamics
    force_eff = force - c_x * x_dot

    # Standard frictionless cart-pole with force -> force_eff
    temp = (force_eff + m * length * theta_dot**2 * sin_t) / total_mass
    num = g * sin_t - cos_t * temp
    den = length * (4.0 / 3.0 - m * cos_t**2 / total_mass)
    theta_ddot = num / den
    x_ddot = temp - (m * length * theta_ddot * cos_t) / total_mass

    # Pivot viscous damping (interpreted as per-second coefficient)
    theta_ddot = theta_ddot - theta_damp * theta_dot

    # Semi-implicit Euler
    x_dot_new = x_dot + dt * x_ddot
    x_new = x + dt * x_dot_new
    theta_dot_new = theta_dot + dt * theta_ddot
    theta_new = theta + dt * theta_dot_new

    # Rail clamp (no bounce)
    x_new_clipped = jnp.clip(x_new, -rail_limit, rail_limit)
    clipped = x_new_clipped != x_new
    x_new = x_new_clipped
    x_dot_new = jnp.where(clipped, jnp.zeros_like(x_dot_new), x_dot_new)

    # Wrap angle to [-pi, pi]
    theta_new = jnp.mod(theta_new + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    return x_new, x_dot_new, theta_new, theta_dot_new


@jax.jit
def get_obs(x: chex.Array, x_dot: chex.Array, theta: chex.Array, theta_dot: chex.Array) -> chex.Array:
    """Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]"""
    return jnp.concatenate([x, x_dot, jnp.cos(theta), jnp.sin(theta), theta_dot], axis=-1)


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
    Smooth reward favoring upright pole, small velocities, and gentle control.
    (Removed unreachable code in original version.)
    """
    r = jnp.cos(theta) - 0.01 * (theta_dot**2) - 0.1 * (x**2 + x_dot**2) - 1e-4 * (force**2)
    return r.squeeze(-1)


class CartPoleEnv:
    """
    JAX-based vectorized cart-pole environment with soft "edge braking"
    via position-dependent viscous damping.
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
        # New damping params (can be tuned at init)
        x_damp_base: float = DEFAULT_PARAMS['x_damp_base'],
        x_damp_edge_k: float = DEFAULT_PARAMS['x_damp_edge_k'],
        x_damp_edge_p: float = DEFAULT_PARAMS['x_damp_edge_p'],
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

        # Damping config
        self.x_damp_base = x_damp_base
        self.x_damp_edge_k = x_damp_edge_k
        self.x_damp_edge_p = x_damp_edge_p
        self.theta_damp = theta_damp

        # Action and observation spaces
        self.action_dim = 1  # Force applied to base
        self.obs_dim = 5  # [x, x_dot, cos(theta), sin(theta), theta_dot]

        # Initialize random key
        self.key = jax.random.PRNGKey(42)

    def reset(self) -> Tuple[chex.Array, CartPoleState]:
        """Reset environment(s) to initial state."""
        self.key, reset_key = jax.random.split(self.key)

        def rand(shape, lo, hi):
            nonlocal reset_key
            reset_key, key = jax.random.split(reset_key)
            return jax.random.uniform(key, shape, minval=lo, maxval=hi)

        x = rand((self.num_envs, 1), -1.0, 1.0)
        theta_dot = rand((self.num_envs, 1), -0.5, 0.5)
        theta = rand((self.num_envs, 1), -jnp.pi, jnp.pi)
        x_dot = rand((self.num_envs, 1), -0.5, 0.5)

        state = CartPoleState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        obs = get_obs(x, x_dot, theta, theta_dot)
        return obs, state

    def step(
        self, state: CartPoleState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, CartPoleState]:
        """Step environment(s) forward."""
        # Clip action to valid range
        action = jnp.asarray(action).reshape(self.num_envs, 1)
        force = jnp.clip(action, -self.max_force, self.max_force)

        # Physics step (soft edge damping inside)
        next_x, next_x_dot, next_theta, next_theta_dot = cartpole_step(
            x=state.x,
            x_dot=state.x_dot,
            theta=state.theta,
            theta_dot=state.theta_dot,
            force=force,
            dt=self.dt,
            g=self.g,
            length=self.length,
            m=self.m,
            M=self.M,
            rail_limit=self.rail_limit,
            x_damp_base=self.x_damp_base,
            x_damp_edge_k=self.x_damp_edge_k,
            x_damp_edge_p=self.x_damp_edge_p,
            theta_damp=self.theta_damp,
        )

        # Clip velocities (kept as safety caps)
        next_x_dot = jnp.clip(next_x_dot, -self.max_base_speed, self.max_base_speed)
        next_theta_dot = jnp.clip(next_theta_dot, -self.max_speed, self.max_speed)

        # Reward & outputs
        reward = reward_fn(state.x, state.x_dot, state.theta, state.theta_dot, force, self.length)
        next_obs = get_obs(next_x, next_x_dot, next_theta, next_theta_dot)
        done = jnp.zeros((self.num_envs,), dtype=bool)

        next_state = CartPoleState(x=next_x, x_dot=next_x_dot, theta=next_theta, theta_dot=next_theta_dot)
        return next_obs, reward, done, next_state
