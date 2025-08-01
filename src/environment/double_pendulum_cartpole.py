from functools import partial
import jax
import chex
import jax.numpy as jnp
from typing import Callable, Tuple


DTYPE = jnp.float64

# Default physical parameters for double pendulum
DEFAULT_PARAMS = {
    'dt': 1 / 200,
    'g': 9.81,
    'length1': 1.2,  # First pendulum length
    'length2': 1.0,  # Second pendulum length
    'M': 2.0,  # Cart mass
    'm1': 0.2,  # First pendulum mass
    'm2': 0.05,  # Second pendulum mass
    'max_base_speed': 6.0,  # TODO reduce
    'max_speed': 8.0,  # TODO reduce
    'max_force': 50.0,  # TODO reduce
    'rail_limit': 10.0,  # base can move between -5 and 5
    'x_damp': 0.02,  # Cart velocity damping
    'theta_damp1': 0.02,  # First pole rotational damping
    'theta_damp2': 0.02,  # Second pole rotational damping
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
    # Gradients
    dLdq = jax.grad(_lagrangian, argnums=0)(positions, velocities, params)  # ∂L/∂q

    # Mass matrix M(q) = ∂/∂v (∂L/∂v)
    Mmat = jax.jacfwd(lambda vv: jax.grad(_lagrangian, 1)(positions, vv, params))(velocities)

    # C(q,v) v term = (∂/∂q ∂L/∂v) v
    C_times_v = jax.jacfwd(lambda qq: jax.grad(_lagrangian, 1)(qq, velocities, params))(positions) @ velocities

    Q = _generalized_forces(velocities, params, force)

    rhs = Q + dLdq - C_times_v
    vdot = jnp.linalg.solve(Mmat, rhs)  # stable, no explicit inverse
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
def reward_fn_old(
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
    ) * 0.5

    return r


@jax.jit
def reward_fn_old_1(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    length1: float,
    length2: float,
    rail_limit: float,
    max_base_speed: float = 10.0,
    max_speed: float = 10.0,
    max_force: float = 100.0,
) -> chex.Array:
    """
    Bounded, cost-style reward:
      r = - (wθ * angle_cost + wv * vel_cost + wx * pos_cost + wu * control_cost + wb * boundary_cost)

    Scales all terms to be dimensionless and softly clipped so nothing explodes.
    The best attainable reward is ~0.
    """

    # --- helpers ---
    def huber(x, k=1.0):
        # Smooth L1; ~0.5*(x/k)^2 for |x|<=k, ~|x|/k - 0.5 otherwise
        ax = jnp.abs(x)
        return jnp.where(ax <= k, 0.5 * (x / k) ** 2, ax / k - 0.5)

    # Angles are already wrapped in your step function; just in case:
    th1 = _wrap_angle(state.theta1)
    th2 = _wrap_angle(state.theta2)

    # --- angle cost (0 at upright, ~2 at inverted) ---
    # Using 1 - cos(theta): smooth, bounded in [0, 2]
    angle_cost = (1.0 - jnp.cos(th1)) + (1.0 - jnp.cos(th2))

    # --- angular velocity cost ---
    th1d_n = state.theta1_dot / max_speed
    th2d_n = state.theta2_dot / max_speed
    ang_vel_cost = huber(th1d_n, k=1.0) + huber(th2d_n, k=1.0)

    # --- cart position & velocity costs ---
    x_n = state.x / rail_limit  # in [-1, 1] normally
    xd_n = state.x_dot / max_base_speed
    pos_cost = huber(x_n, k=1.0)  # softly penalize |x|>0
    base_vel_cost = huber(xd_n, k=1.0)

    # --- control effort (scaled) ---
    u_n = force / max_force
    control_cost = 0.5 * (u_n**2)  # simple quadratic, bounded by 0.5

    # --- boundary barrier (smooth, only near rails) ---
    # Kicks in when |x| > rail_limit - margin; softplus keeps it smooth & bounded per step
    margin = 0.5
    over = jnp.maximum(jnp.abs(state.x) - (rail_limit - margin), 0.0)
    boundary_cost = jnp.log1p(jnp.exp(10.0 * (over / margin))) / 10.0  # softplus with scale

    # --- weights (tune here) ---
    w_theta = 1.0  # uprightness is primary
    w_vel = 0.15  # damp angular speed
    w_pos = 0.25  # keep cart centered
    w_bvel = 0.05  # don't rush the cart
    w_u = 0.005  # gentle effort regularizer
    w_bound = 1.0  # strongly discourage rail contact

    total_cost = (
        w_theta * angle_cost
        + w_vel * ang_vel_cost
        + w_pos * pos_cost
        + w_bvel * base_vel_cost
        + w_u * control_cost
        + w_bound * boundary_cost
    )

    # Reward is negative cost; clip to a tidy range to stabilize Q-targets
    r = -total_cost
    r = jnp.clip(r, -5.0, 0.0)
    return r


@jax.jit
def reward_fn_old_2(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    rail_limit: float,
    max_base_speed: float = 10.0,
    max_speed: float = 10.0,
    max_force: float = 100.0,
) -> chex.Array:
    """Bounded reward in [-5, 0]; best is 0. Uses NEXT state to avoid action-spike exploits."""

    # --- helpers ---
    def huber(x, k=1.0):
        ax = jnp.abs(x)
        return jnp.where(ax <= k, 0.5 * (x / k) ** 2, ax / k - 0.5)

    th1 = _wrap_angle(state.theta1)
    th2 = _wrap_angle(state.theta2)

    # Uprightness (bounded 0..2 per link)
    upright_cost = (1 - jnp.cos(th1)) + (1 - jnp.cos(th2))

    # Keep both links aligned around upright (penalize relative hinge motion)
    relative_cost = 0.5 * (1 - jnp.cos(th1 - th2))  # 0 when aligned

    # Angular velocity (normalized + Huber)
    th1d = huber(state.theta1_dot / max_speed, k=1.0)
    th2d = huber(state.theta2_dot / max_speed, k=1.0)
    ang_vel_cost = th1d + th2d

    # Cart position/velocity (normalized + Huber)
    x_cost = huber(state.x / rail_limit, k=1.0)
    xd_cost = huber(state.x_dot / max_base_speed, k=1.0)

    # Control (normalized)
    u_cost = 0.5 * (force / max_force) ** 2

    # Smooth rail barrier (activates in last 0.4 m)
    margin = 0.4
    over = jnp.maximum(jnp.abs(state.x) - (rail_limit - margin), 0.0) / margin
    boundary_cost = jnp.log1p(jnp.exp(8.0 * over)) / 8.0

    # Weights
    w_upright = 1.8
    w_rel = 0.6
    w_angv = 0.2
    w_x = 0.1
    w_xd = 0.05
    w_u = 0.005
    w_bound = 1.5

    cost = (
        w_upright * upright_cost
        + w_rel * relative_cost
        + w_angv * ang_vel_cost
        + w_x * x_cost
        + w_xd * xd_cost
        + w_u * u_cost
        + w_bound * boundary_cost
    )

    r = -cost
    return jnp.clip(r, -5.0, 0.0)


@jax.jit
def reward_fn_old_3(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    rail_limit: float,
    max_base_speed: float = 10.0,
    max_speed: float = 10.0,
    max_force: float = 100.0,
) -> chex.Array:
    def huber(x, k=2.0):  # wider k so typical resets aren’t penalized too hard
        ax = jnp.abs(x)
        return jnp.where(ax <= k, 0.5 * (x / k) ** 2, ax / k - 0.5)

    th1 = _wrap_angle(state.theta1)
    th2 = _wrap_angle(state.theta2)

    upright_cost = (1 - jnp.cos(th1)) + (1 - jnp.cos(th2))  # ∈ [0, 4]
    relative_cost = 0.5 * (1 - jnp.cos(th1 - th2))  # ∈ [0, 1]

    ang_vel_cost = huber(state.theta1_dot / max_speed) + huber(state.theta2_dot / max_speed)

    x_cost = huber(state.x / rail_limit)
    xd_cost = huber(state.x_dot / max_base_speed)

    u_cost = 0.5 * (force / max_force) ** 2

    # gentler barrier; activates in last 0.6 m
    margin = 0.6
    over = jnp.maximum(jnp.abs(state.x) - (rail_limit - margin), 0.0) / margin
    boundary_cost = jnp.log1p(jnp.exp(5.0 * over)) / 5.0

    # ↓ scales reduced ~×0.5 from previous suggestion
    w_upright = 0.9
    w_rel = 0.3
    w_angv = 0.1
    w_x = 0.15
    w_xd = 0.03
    w_u = 0.003
    w_bound = 0.8

    cost = (
        w_upright * upright_cost
        + w_rel * relative_cost
        + w_angv * ang_vel_cost
        + w_x * x_cost
        + w_xd * xd_cost
        + w_u * u_cost
        + w_bound * boundary_cost
    )

    # Soft squash instead of hard clip; keeps gradients when cost is big
    r = -10.0 * jnp.tanh(cost / 10.0)  # ∈ (-10, 0)
    return r


@jax.jit
def reward_fn(
    state: DoublePendulumCartPoleState,
    force: chex.Array,
    rail_limit: float,
    max_base_speed: float,
    max_speed: float,
    max_force: float,
) -> chex.Array:
    def huber(x, k=2.0):
        ax = jnp.abs(x)
        return jnp.where(ax <= k, 0.5 * (x / k) ** 2, ax / k - 0.5)

    th1 = _wrap_angle(state.theta1)
    th2 = _wrap_angle(state.theta2)

    upright_cost = (1 - jnp.cos(th1)) + (1 - jnp.cos(th2))  # [0,4]
    rel_cost = 0.5 * (1 - jnp.cos(th1 - th2))  # [0,1]
    angv_cost = huber(state.theta1_dot / max_speed) + huber(state.theta2_dot / max_speed)
    x_cost = huber(state.x / rail_limit)
    xd_cost = huber(state.x_dot / max_base_speed)
    u_cost = 0.5 * (force / max_force) ** 2

    margin = 0.6
    over = jnp.maximum(jnp.abs(state.x) - (rail_limit - margin), 0.0) / margin
    boundary_cost = jnp.log1p(jnp.exp(5.0 * over)) / 5.0

    # modest weights so typical resets aren't saturated
    wθ, wrel, wω, wx, wxd, wu, wb = 1.5, 0.3, 0.1, 0.15, 0.03, 0.003, 2.5
    cost = (
        wθ * upright_cost
        + wrel * rel_cost
        + wω * angv_cost
        + wx * x_cost
        + wxd * xd_cost
        + wu * u_cost
        + wb * boundary_cost
    )

    # soft squash keeps gradients at large cost, but bounds scale
    return -5.0 * jnp.tanh(cost / 5.0)


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
