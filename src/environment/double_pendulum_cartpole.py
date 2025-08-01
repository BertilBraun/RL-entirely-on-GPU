from functools import partial
import jax
import chex
import jax.numpy as jnp
from typing import Callable, Tuple


DTYPE = jnp.float64

# Default physical parameters for double pendulum
# Parameters are based on: https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/inverted_double_pendulum.xml
DEFAULT_PARAMS = {
    'dt': 0.01,
    'g': 9.81,
    'length1': 2.0,  # First pendulum length
    'length2': 1.8,  # Second pendulum length
    'M': 2.0,  # Cart mass
    'm1': 0.4,  # First pendulum mass
    'm2': 0.4,  # Second pendulum mass
    'max_base_speed': 8.0,  # TODO reduce
    'max_speed': 8.0,  # TODO reduce
    'max_force': 20.0,  # TODO reduce
    'rail_limit': 10.0,  # base can move between -5 and 5
    'x_damp': 0.05,  # Cart velocity damping
    'theta_damp1': 0.05,  # First pole rotational damping
    'theta_damp2': 0.05,  # Second pole rotational damping
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


# -------------------------------
# Params & State
# -------------------------------


@chex.dataclass(frozen=True)
class Params:
    dt: float
    g: float
    length1: float
    length2: float
    m1: float
    m2: float
    M: float
    x_damp: float
    theta_damp1: float
    theta_damp2: float


# -------------------------------
# Lagrangian-based EoM (autodiff)
# -------------------------------


def _lagrangian(positions: chex.Array, velocities: chex.Array, params: Params) -> chex.Array:
    """
    q = [x, th1, th2], v = [xd, th1d, th2d]
    Angles measured from upright (0 = up). Positive y downward.
    """
    x, th1, th2 = positions  # type: ignore
    xd, th1d, th2d = velocities  # type: ignore
    l1, l2 = params.length1, params.length2
    m1, m2, M = params.m1, params.m2, params.M
    g = -params.g

    # Link COM positions in cart frame (pivot at cart, y down)
    # Using full-length rods; if your COM is at l/2, replace l -> l/2 below.
    y1 = -l1 * jnp.cos(th1)
    y2 = y1 - l2 * jnp.cos(th2)

    # Velocities of link COMs in world frame
    v1x = xd + l1 * jnp.cos(th1) * th1d
    v1y = l1 * jnp.sin(th1) * th1d
    v2x = v1x + l2 * jnp.cos(th2) * th2d
    v2y = v1y + l2 * jnp.sin(th2) * th2d

    T = 0.5 * M * xd**2 + 0.5 * m1 * (v1x**2 + v1y**2) + 0.5 * m2 * (v2x**2 + v2y**2)

    V = m1 * g * y1 + m2 * g * y2  # y down ⇒ upright is high potential, unstable

    return T - V  # L = T - V


def _generalized_forces(velocities: chex.Array, params: Params, force: chex.Array) -> chex.Array:
    """Q = [Fx_on_cart, τ1, τ2] with joint Rayleigh damping."""
    xd, th1d, th2d = velocities  # type: ignore
    # Cart actuation (force along x), viscous damping at joints
    return jnp.array([force - params.x_damp * xd, -params.theta_damp1 * th1d, -params.theta_damp2 * th2d], dtype=DTYPE)


def _accelerations_single(
    positions: chex.Array, velocities: chex.Array, params: Params, force: chex.Array
) -> chex.Array:
    """
    Returns vdot solving:  d/dt(∂L/∂v) - ∂L/∂q = Q  ⇒  M(q) vdot = Q + ∂L/∂q - (∂/∂q ∂L/∂v) v
    """
    # Light bounds checking to prevent extreme values
    positions = jnp.clip(positions, -50.0, 50.0)
    velocities = jnp.clip(velocities, -50.0, 50.0)

    # Gradients
    dLdq = jax.grad(_lagrangian, argnums=0)(positions, velocities, params)  # ∂L/∂q

    # Mass matrix M(q) = ∂/∂v (∂L/∂v)
    Mmat = jax.jacfwd(lambda vv: jax.grad(_lagrangian, 1)(positions, vv, params))(velocities)

    # C(q,v) v term = (∂/∂q ∂L/∂v) v
    C_times_v = jax.jacfwd(lambda qq: jax.grad(_lagrangian, 1)(qq, velocities, params))(positions) @ velocities

    Q = _generalized_forces(velocities, params, force)

    rhs = Q + dLdq - C_times_v

    # Simple regularization for numerical stability
    Mmat_reg = Mmat + 1e-8 * jnp.eye(Mmat.shape[0])
    vdot = jnp.linalg.solve(Mmat_reg, rhs)

    # Light bounds on output
    vdot = jnp.clip(vdot, -100.0, 100.0)

    return vdot


def _wrap_angle(angle: chex.Array) -> chex.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _step_single(
    x: chex.Array,
    x_dot: chex.Array,
    theta1: chex.Array,
    theta1_dot: chex.Array,
    theta2: chex.Array,
    theta2_dot: chex.Array,
    force: chex.Array,
    params: Params,
) -> DoublePendulumCartPoleState:
    """
    Semi-implicit (symplectic) Euler:
      v_{t+1} = v_t + dt * vdot(q_t, v_t)
      q_{t+1} = q_t + dt * v_{t+1}
    """
    positions = jnp.array([x, theta1, theta2], dtype=DTYPE)
    velocities = jnp.array([x_dot, theta1_dot, theta2_dot], dtype=DTYPE)

    vdot = _accelerations_single(positions, velocities, params, force)

    velocities_new = velocities + params.dt * vdot
    positions_new = positions + params.dt * velocities_new

    x_new, th1_new, th2_new = positions_new
    xd_new, th1d_new, th2d_new = velocities_new

    return DoublePendulumCartPoleState(
        x=x_new,
        x_dot=xd_new,
        theta1=_wrap_angle(th1_new),
        theta1_dot=th1d_new,
        theta2=_wrap_angle(th2_new),
        theta2_dot=th2d_new,
    )


# -------------------------------
# Batched, JIT-compiled step
# -------------------------------


def make_batched_step(
    params: Params,
) -> Callable[[DoublePendulumCartPoleState, chex.Array], DoublePendulumCartPoleState]:
    """
    Returns a compiled function that advances a batch of environments in parallel.

    Input shapes:
      - State fields: (batch,) or () scalars; all must share the same leading shape
      - u (force):    (batch,)   (cart force per env)

    Output:
      - next State with same leading shape
    """
    # Vectorize _step_single over the leading axis
    vmap_step = jax.vmap(
        _step_single,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None),
    )

    # JIT compile; donate state to reduce memory traffic
    @jax.jit
    def step_batched(state: DoublePendulumCartPoleState, force: chex.Array) -> DoublePendulumCartPoleState:
        return vmap_step(
            state.x,
            state.x_dot,
            state.theta1,
            state.theta1_dot,
            state.theta2,
            state.theta2_dot,
            force,
            params,
        )

    return step_batched


@jax.jit
def get_obs(
    state: DoublePendulumCartPoleState, rail_limit: float, max_base_speed: float, max_speed: float
) -> chex.Array:
    """
    Observation: [x, x_dot, cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot]
    """
    return jnp.stack(
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
        axis=1,
        dtype=DTYPE,
    )


@jax.jit
def reward_fn(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    rail_limit: float,
    max_base_speed: float,
    max_speed: float,
    max_force: float,
) -> chex.Array:
    """
    Balanced reward function for stable double pendulum learning.

    Design principles:
    - Symmetric (no left/right bias)
    - Smooth gradients for good learning
    - Balanced component scales
    - Progressive difficulty
    """

    # === UPRIGHTNESS REWARD (0-4 total) ===
    # Softer exponentials for smoother gradients
    upright_1 = 2.0 * jnp.exp(-0.5 * state.theta1**2)  # Less steep
    upright_2 = 2.0 * jnp.exp(-0.5 * state.theta2**2)

    # === STABILITY REWARD (0-2 total) ===
    # Reward low angular velocities
    stable_1 = 1.0 * jnp.exp(-0.05 * state.theta1_dot**2)  # Even softer
    stable_2 = 1.0 * jnp.exp(-0.05 * state.theta2_dot**2)

    # === POSITION REWARD (0-1 total) ===
    # Gentle preference for center, but not too strong
    position_reward = 1.0 * jnp.exp(-0.1 * state.x**2)

    # === VELOCITY CONTROL (0-1 total) ===
    # Reward controlled cart movement
    velocity_reward = 1.0 * jnp.exp(-0.01 * state.x_dot**2)

    # === PROGRESS BONUS (0-2 total) ===
    # Bonus for achieving partial goals
    partially_upright = (jnp.abs(state.theta1) < 1.0) & (jnp.abs(state.theta2) < 1.0)
    well_controlled = (jnp.abs(state.theta1_dot) < 5.0) & (jnp.abs(state.theta2_dot) < 5.0)
    progress_bonus = jnp.where(partially_upright, 1.0, 0.0) + jnp.where(well_controlled, 1.0, 0.0)

    # === BOUNDARY MANAGEMENT ===
    # Much gentler boundary handling - gradual penalty, not cliff
    boundary_factor = jnp.maximum(0.0, (jnp.abs(state.x) - rail_limit * 0.6) / (rail_limit * 0.4))
    boundary_penalty = -2.0 * boundary_factor  # Max -2.0, gradual

    # === ENERGY PENALTY ===
    # Small penalty for excessive force to encourage efficiency
    energy_penalty = -0.1 * (force / max_force) ** 2

    # Total: 0 to ~11 when perfect, with smooth gradients everywhere
    total_reward = (
        upright_1
        + upright_2
        + stable_1
        + stable_2
        + position_reward
        + velocity_reward
        + progress_bonus
        + boundary_penalty
        + energy_penalty
    )

    # Ensure reasonable bounds
    return jnp.clip(total_reward, -5.0, 15.0)


@jax.jit
def is_done(state: DoublePendulumCartPoleState, rail_limit: float) -> chex.Array:
    """Episode is done when cart hits boundary."""
    return (jnp.abs(state.x) >= rail_limit).reshape(-1)


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
        M: float = DEFAULT_PARAMS['M'],
        m1: float = DEFAULT_PARAMS['m1'],
        m2: float = DEFAULT_PARAMS['m2'],
        x_damp: float = DEFAULT_PARAMS['x_damp'],
        theta_damp1: float = DEFAULT_PARAMS['theta_damp1'],
        theta_damp2: float = DEFAULT_PARAMS['theta_damp2'],
    ) -> None:
        assert num_envs > 0, 'num_envs must be at least 1'

        # Action and observation spaces
        self.action_dim = 1  # Force applied to cart
        self.obs_dim = 8  # [x, x_dot, cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot]

        self.num_envs = num_envs
        self.max_base_speed = max_base_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.rail_limit = rail_limit

        self.params = Params(
            dt=dt,
            g=g,
            length1=length1,
            length2=length2,
            M=M,
            m1=m1,
            m2=m2,
            x_damp=x_damp,
            theta_damp1=theta_damp1,
            theta_damp2=theta_damp2,
        )

        self._step = make_batched_step(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, reset_key: chex.PRNGKey) -> Tuple[chex.Array, DoublePendulumCartPoleState]:
        """Reset environment(s) to initial state."""

        def rand(shape: Tuple[int, ...], lo: float, hi: float, key: chex.PRNGKey) -> chex.Array:
            key, sub_key = jax.random.split(key)
            return jax.random.uniform(sub_key, shape, minval=lo, maxval=hi, dtype=DTYPE)

        k1, k2, k3, k4, k5, k6 = jax.random.split(reset_key, 6)

        # Initialize cart position and velocity
        x = rand((self.num_envs,), -1.0, 1.0, k1)
        x_dot = rand((self.num_envs,), -0.5, 0.5, k2)

        # Initialize first pendulum (hanging down to slightly off vertical)
        theta1 = rand((self.num_envs,), -jnp.pi, jnp.pi, k3)
        theta1_dot = rand((self.num_envs,), -0.5, 0.5, k4)

        # Initialize second pendulum (hanging down to slightly off vertical)
        theta2 = rand((self.num_envs,), -jnp.pi, jnp.pi, k5)
        theta2_dot = rand((self.num_envs,), -0.5, 0.5, k6)

        state = DoublePendulumCartPoleState(
            x=x,
            x_dot=x_dot,
            theta1=theta1,
            theta1_dot=theta1_dot,
            theta2=theta2,
            theta2_dot=theta2_dot,
        )
        obs = get_obs(state, self.rail_limit, self.max_base_speed, self.max_speed)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DoublePendulumCartPoleState, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, DoublePendulumCartPoleState]:
        """Step environment(s) forward."""
        # Clip action to valid range
        action = jnp.asarray(action, dtype=DTYPE).reshape(self.num_envs)
        force = jnp.clip(action, -self.max_force, self.max_force)

        # Physics step
        next_state = self._step(state, force)

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
        reward = reward_fn(
            next_state,  # TODO use state instead of next_state
            force=force,
            rail_limit=self.rail_limit,
            max_base_speed=self.max_base_speed,
            max_speed=self.max_speed,
            max_force=self.max_force,
        )
        next_obs = get_obs(next_state, self.rail_limit, self.max_base_speed, self.max_speed)

        return next_obs, reward, done, next_state
