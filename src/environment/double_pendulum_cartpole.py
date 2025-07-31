from functools import partial
import jax
import chex
import jax.numpy as jnp
from typing import Tuple


# Default physical parameters for double pendulum
DEFAULT_PARAMS = {
    'dt': 1 / 60,
    'g': 9.81,
    'length1': 1.0,  # First pendulum length
    'length2': 1.0,  # Second pendulum length
    'm1': 0.01,  # First pendulum mass
    'm2': 0.01,  # Second pendulum mass
    'M': 1.0,  # Cart mass
    'max_speed': 4.0,
    'max_base_speed': 3.0,
    'max_force': 15.0,
    'rail_limit': 4.0,  # base can move between -4 and 4
    'theta_damp1': 0.00,  # First pole rotational damping
    'theta_damp2': 0.00,  # Second pole rotational damping
}


@chex.dataclass
class DoublePendulumCartPoleState:
    """State representation for double pendulum cart-pole environment."""

    x: chex.Array  # Cart position
    x_dot: chex.Array  # Cart velocity
    theta1: chex.Array  # First pendulum angle (from vertical)
    theta1_dot: chex.Array  # First pendulum angular velocity
    theta2: chex.Array  # Second pendulum angle (from vertical)
    theta2_dot: chex.Array  # Second pendulum angular velocity


@jax.jit
def double_pendulum_cartpole_step(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    dt: float,
    g: float,
    length1: float,
    length2: float,
    m1: float,
    m2: float,
    M: float,
    rail_limit: float,
    theta_damp1: float,
    theta_damp2: float,
) -> DoublePendulumCartPoleState:
    """
    Double pendulum on cart dynamics with complex coupled equations.
    Based on Lagrangian mechanics for double pendulum on movable cart.
    """
    # Extract current state
    x, x_dot = state.x, state.x_dot
    theta1, theta1_dot = state.theta1, state.theta1_dot
    theta2, theta2_dot = state.theta2, state.theta2_dot

    # Trigonometric terms
    cos1 = jnp.cos(theta1)
    sin1 = jnp.sin(theta1)
    cos2 = jnp.cos(theta2)
    sin2 = jnp.sin(theta2)
    cos12 = jnp.cos(jnp.subtract(theta1, theta2))
    sin12 = jnp.sin(jnp.subtract(theta1, theta2))

    # Total mass
    total_mass = M + m1 + m2

    # Mass matrix elements (for the system: [x_ddot, theta1_ddot, theta2_ddot])
    # M11 = total_mass
    M12 = (m1 + m2) * length1 * cos1
    M13 = m2 * length2 * cos2
    # M21 = (m1 + m2) * length1 * cos1
    M22 = (m1 + m2) * length1**2
    M23 = m2 * length1 * length2 * cos12
    # M31 = m2 * length2 * cos2
    M32 = m2 * length1 * length2 * cos12
    M33 = m2 * length2**2

    # Right hand side vector elements
    # Gravitational and centrifugal terms
    f1 = force + (m1 + m2) * length1 * theta1_dot**2 * sin1 + m2 * length2 * theta2_dot**2 * sin2
    f2 = -(m1 + m2) * g * length1 * sin1 + m2 * length1 * length2 * theta2_dot**2 * sin12
    f3 = -m2 * g * length2 * sin2 - m2 * length1 * length2 * theta1_dot**2 * sin12

    # Apply damping
    f2 -= theta_damp1 * theta1_dot * (m1 + m2) * length1**2
    f3 -= theta_damp2 * theta2_dot * m2 * length2**2

    # Solve the linear system M * [x_ddot, theta1_ddot, theta2_ddot]^T = [f1, f2, f3]^T
    # Using manual inversion of 3x3 matrix for efficiency

    # Determinant of mass matrix
    det = total_mass * M22 * M33 + M12 * M23 * M13 * 2 - total_mass * M23**2 - M12**2 * M33 - M13**2 * M22

    # Inverse mass matrix elements (only what we need)
    inv_M11 = (M22 * M33 - M23**2) / det
    inv_M12 = (M13 * M23 - M12 * M33) / det
    inv_M13 = (M12 * M23 - M13 * M22) / det
    inv_M21 = inv_M12  # Symmetric
    inv_M22 = (total_mass * M33 - M13**2) / det
    inv_M23 = (M12 * M13 - total_mass * M23) / det
    inv_M31 = inv_M13  # Symmetric
    inv_M32 = inv_M23  # Symmetric
    inv_M33 = (total_mass * M22 - M12**2) / det

    # Compute accelerations
    x_ddot = inv_M11 * f1 + inv_M12 * f2 + inv_M13 * f3
    theta1_ddot = inv_M21 * f1 + inv_M22 * f2 + inv_M23 * f3
    theta2_ddot = inv_M31 * f1 + inv_M32 * f2 + inv_M33 * f3

    # Semi-implicit Euler integration
    x_dot_new = x_dot + dt * x_ddot
    x_new = x + dt * x_dot_new

    theta1_dot_new = theta1_dot + dt * theta1_ddot
    theta1_new = theta1 + dt * theta1_dot_new

    theta2_dot_new = theta2_dot + dt * theta2_ddot
    theta2_new = theta2 + dt * theta2_dot_new

    # Wrap angles to [-pi, pi]
    theta1_new = jnp.mod(theta1_new + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    theta2_new = jnp.mod(theta2_new + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    return DoublePendulumCartPoleState(
        x=x_new,
        x_dot=x_dot_new,
        theta1=theta1_new,
        theta1_dot=theta1_dot_new,
        theta2=theta2_new,
        theta2_dot=theta2_dot_new,
    )


@jax.jit
def get_obs(
    state: DoublePendulumCartPoleState, rail_limit: float, max_base_speed: float, max_speed: float
) -> chex.Array:
    """
    Observation: [x, x_dot, cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot]
    """
    return jnp.concatenate(
        [
            state.x / rail_limit,  # Normalize position to [-1, 1]
            state.x_dot / max_base_speed,  # Normalize base velocity to [-1, 1]
            jnp.cos(state.theta1),  # Cosine of first pendulum angle
            jnp.sin(state.theta1),  # Sine of first pendulum angle
            state.theta1_dot / max_speed,  # Normalize first pendulum angular velocity
            jnp.cos(state.theta2),  # Cosine of second pendulum angle
            jnp.sin(state.theta2),  # Sine of second pendulum angle
            state.theta2_dot / max_speed,  # Normalize second pendulum angular velocity
        ],
        axis=-1,
    )


@jax.jit
def reward_fn(
    state: DoublePendulumCartPoleState, force: chex.Array, length1: float, length2: float, rail_limit: float
) -> chex.Array:
    """
    Reward function for double pendulum encouraging both pendulums to be upright.
    """
    # Reward for both pendulums being upright
    upright1 = 2.0 * jnp.exp(-(state.theta1**2) / (0.12**2))
    upright2 = 2.0 * jnp.exp(-(state.theta2**2) / (0.12**2))

    # Cosine rewards (smooth reward for being near upright)
    cos_reward1 = jnp.cos(state.theta1)
    cos_reward2 = jnp.cos(state.theta2)

    # Penalties for high velocities and position
    position_penalty = 0.05 * (state.x**2)
    velocity_penalties = 0.01 * (state.x_dot**2) + 0.01 * (state.theta1_dot**2) + 0.01 * (state.theta2_dot**2)

    # Control effort penalty
    control_penalty = 1e-4 * (force**2)

    # Boundary penalty
    boundary_penalty = jnp.where(jnp.abs(state.x) >= rail_limit - 0.5, 3, 0)

    r = (
        cos_reward1
        + cos_reward2
        + upright1
        + upright2
        - position_penalty
        - velocity_penalties
        - control_penalty
        - boundary_penalty
    )

    return r.squeeze(-1)


@jax.jit
def is_done(state: DoublePendulumCartPoleState, rail_limit: float) -> chex.Array:
    """Episode is done when cart hits boundary."""
    return (jnp.abs(state.x) >= rail_limit).squeeze(-1)


class DoublePendulumCartPoleEnv:
    """
    JAX-based vectorized double pendulum cart-pole environment with complex coupled physics.
    Episode ends when cart hits rails.
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
        length1: float = DEFAULT_PARAMS['length1'],
        length2: float = DEFAULT_PARAMS['length2'],
        m1: float = DEFAULT_PARAMS['m1'],
        m2: float = DEFAULT_PARAMS['m2'],
        M: float = DEFAULT_PARAMS['M'],
        theta_damp1: float = DEFAULT_PARAMS['theta_damp1'],
        theta_damp2: float = DEFAULT_PARAMS['theta_damp2'],
    ) -> None:
        assert num_envs > 0, 'num_envs must be at least 1'

        self.num_envs = num_envs
        self.max_base_speed = max_base_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.rail_limit = rail_limit
        self.dt = dt
        self.g = g
        self.length1 = length1
        self.length2 = length2
        self.m1 = m1
        self.m2 = m2
        self.M = M
        self.theta_damp1 = theta_damp1
        self.theta_damp2 = theta_damp2

        # Action and observation spaces
        self.action_dim = 1  # Force applied to cart
        self.obs_dim = 8  # [x, x_dot, cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, reset_key: chex.PRNGKey) -> Tuple[chex.Array, DoublePendulumCartPoleState]:
        """Reset environment(s) to initial state."""

        def rand(shape: Tuple[int, ...], lo: float, hi: float, key: chex.PRNGKey) -> chex.Array:
            key, sub_key = jax.random.split(key)
            return jax.random.uniform(sub_key, shape, minval=lo, maxval=hi)

        keys = jax.random.split(reset_key, 8)

        # Initialize cart position and velocity
        x = rand((self.num_envs, 1), -1.0, 1.0, keys[0])
        x_dot = rand((self.num_envs, 1), -0.5, 0.5, keys[1])

        # Initialize first pendulum (hanging down to slightly off vertical)
        theta1 = rand((self.num_envs, 1), jnp.pi / 3, 2 * jnp.pi / 3, keys[2])
        negative_theta1 = rand((self.num_envs, 1), 0, 1, keys[3]) > 0.5
        theta1 = jnp.where(negative_theta1, jnp.negative(theta1), theta1)
        theta1_dot = rand((self.num_envs, 1), -0.5, 0.5, keys[4])

        # Initialize second pendulum (hanging down to slightly off vertical)
        theta2 = rand((self.num_envs, 1), jnp.pi / 3, 2 * jnp.pi / 3, keys[5])
        negative_theta2 = rand((self.num_envs, 1), 0, 1, keys[6]) > 0.5
        theta2 = jnp.where(negative_theta2, jnp.negative(theta2), theta2)
        theta2_dot = rand((self.num_envs, 1), -0.5, 0.5, keys[7])

        state = DoublePendulumCartPoleState(
            x=x, x_dot=x_dot, theta1=theta1, theta1_dot=theta1_dot, theta2=theta2, theta2_dot=theta2_dot
        )
        obs = get_obs(state, self.rail_limit, self.max_base_speed, self.max_speed)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DoublePendulumCartPoleState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, DoublePendulumCartPoleState]:
        """Step environment(s) forward."""
        # Clip action to valid range
        action = jnp.asarray(action).reshape(self.num_envs, 1)
        force = jnp.clip(action, -self.max_force, self.max_force)

        # Physics step
        next_state = double_pendulum_cartpole_step(
            state=state,
            force=force,
            dt=self.dt,
            g=self.g,
            length1=self.length1,
            length2=self.length2,
            m1=self.m1,
            m2=self.m2,
            M=self.M,
            rail_limit=self.rail_limit,
            theta_damp1=self.theta_damp1,
            theta_damp2=self.theta_damp2,
        )

        # Clip velocities (safety caps)
        next_state = DoublePendulumCartPoleState(
            x=next_state.x,
            x_dot=jnp.clip(next_state.x_dot, -self.max_base_speed, self.max_base_speed),
            theta1=next_state.theta1,
            theta1_dot=jnp.clip(next_state.theta1_dot, -self.max_speed, self.max_speed),
            theta2=next_state.theta2,
            theta2_dot=jnp.clip(next_state.theta2_dot, -self.max_speed, self.max_speed),
        )

        # Check if episode is done (hit boundary)
        done = is_done(next_state, self.rail_limit)

        # Reward & outputs
        reward = reward_fn(state, force, self.length1, self.length2, self.rail_limit)
        next_obs = get_obs(next_state, self.rail_limit, self.max_base_speed, self.max_speed)

        return next_obs, reward, done, next_state
