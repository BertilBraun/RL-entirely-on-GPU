"""
Proximal Policy Optimization (PPO) algorithm implementation using JAX.
"""

from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
import chex
import optax
from typing import NamedTuple, Tuple, Any

from algorithms.base import Algorithm, AlgorithmInfo
from networks.actor import ActorNetwork
from networks.value import ValueNetwork
from config import DTYPE
from training.data_structures import PPOConfig


class PPOState(NamedTuple):
    """State of the PPO algorithm."""

    actor_params: chex.ArrayTree
    critic_params: chex.ArrayTree

    actor_opt_state: chex.ArrayTree
    critic_opt_state: chex.ArrayTree

    def save_model(self, step: int) -> None:
        """Save model parameters."""
        # TODO: Implement checkpointing
        pass

    def try_load(self) -> PPOState:
        """Try to load model parameters."""
        # TODO: Implement checkpoint loading
        return self


@chex.dataclass
class PolicyInfo:
    """Info from policy updates."""

    policy_loss: chex.Array
    entropy: chex.Array
    kl_divergence: chex.Array
    clip_fraction: chex.Array


@chex.dataclass
class ValueInfo:
    """Info from value function updates."""

    value_loss: chex.Array
    value_mean: chex.Array
    explained_variance: chex.Array


@chex.dataclass
class PPOInfo(AlgorithmInfo):
    """Info from PPO training."""

    policy_info: PolicyInfo
    value_info: ValueInfo
    total_loss: chex.Array


@chex.dataclass
class PPOBatch:
    """PPO training batch with advantages and returns."""

    obs: chex.Array  # (batch_size, obs_dim)
    actions: chex.Array  # (batch_size, action_dim)
    old_log_probs: chex.Array  # (batch_size,)
    advantages: chex.Array  # (batch_size,)
    returns: chex.Array  # (batch_size,)


class PPO(Algorithm):
    """
    Proximal Policy Optimization algorithm implementation.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_action: float, config: PPOConfig = PPOConfig()) -> None:
        super().__init__(obs_dim, action_dim, max_action)
        self.config = config

        # Create networks
        self.actor = ActorNetwork(hidden_dims=config.actor_hidden_dims, action_dim=action_dim, max_action=max_action)
        self.critic = ValueNetwork(hidden_dims=config.critic_hidden_dims)

        # Create optimizers
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

    def init_state(self, key: chex.PRNGKey) -> PPOState:
        """Initialize PPO state with network parameters and optimizers."""
        key_actor, key_critic = jax.random.split(key, 2)

        # Initialize network parameters
        dummy_obs = jnp.zeros((1, self.obs_dim), dtype=DTYPE)

        actor_params = self.actor.init(key_actor, dummy_obs)
        critic_params = self.critic.init(key_critic, dummy_obs)

        # Initialize optimizers
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)

        return PPOState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
        )

    @property
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        return 'PPO'

    @property
    def requires_replay_buffer(self) -> bool:
        """Return True if algorithm uses replay buffer (off-policy), False if episode buffer (on-policy)."""
        return False

    @partial(jax.jit, static_argnums=0)
    def collect_rollout_data(self, state: PPOState, obs: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, dict]:
        """Collect PPO rollout data - action, log_prob, and value."""
        action, log_prob, value = self.get_action_and_value(state, obs, key)
        rollout_data = {
            'log_prob': log_prob,
            'value': value,
        }
        return action, rollout_data

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
        """Update PPO episode buffer with step data."""
        from algorithms.episode_buffer import EpisodeBuffer

        log_prob = rollout_data['log_prob']
        value = rollout_data['value']

        # Squeeze to match buffer expectations
        value_squeezed = value.squeeze(-1) if value.ndim > 1 else value
        log_prob_squeezed = log_prob.squeeze(-1) if log_prob.ndim > 1 else log_prob

        return EpisodeBuffer.add_step(buffer_state, obs, action, reward, value_squeezed, log_prob_squeezed, done)

    def _compute_gae_advantages(
        self, rewards: chex.Array, values: chex.Array, dones: chex.Array, num_envs: int
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute advantages using Generalized Advantage Estimation (GAE) - only on valid data."""
        # Reshape to (steps, num_envs) for proper GAE computation
        # Note: This assumes data is laid out sequentially: [step0_env0, step0_env1, ..., step1_env0, step1_env1, ...]
        total_elements = rewards.shape[0]
        steps = total_elements // num_envs

        rewards_2d = rewards.reshape(steps, num_envs)
        values_2d = values.reshape(steps, num_envs)
        dones_2d = dones.reshape(steps, num_envs)

        # Bootstrap with final values (zero for terminated episodes)
        bootstrap_values = jnp.zeros(num_envs, dtype=DTYPE)
        values_with_bootstrap = jnp.concatenate([values_2d, bootstrap_values[None, :]], axis=0)

        # Compute TD errors (delta)
        deltas = rewards_2d + self.config.gamma * values_with_bootstrap[1:] * (1 - dones_2d) - values_2d

        # Compute GAE advantages using scan (reverse iteration)
        def gae_step(gae: chex.Array, inputs: Tuple[chex.Array, chex.Array]) -> Tuple[chex.Array, chex.Array]:
            delta, done = inputs
            gae = delta + self.config.gamma * self.config.gae_lambda * gae * (1 - done)
            return gae, gae

        # Process in reverse order (from last step to first)
        _, advantages_reversed = jax.lax.scan(
            gae_step,
            init=jnp.zeros(num_envs, dtype=DTYPE),
            xs=(jnp.flip(deltas, axis=0), jnp.flip(dones_2d, axis=0)),
            reverse=False,
        )

        # Flip back to get advantages in correct order and flatten
        advantages = jnp.flip(advantages_reversed, axis=0).reshape(-1)

        # Compute returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def prepare_update_batch(self, buffer_state: Any, key: chex.PRNGKey, batch_size: int) -> PPOBatch:
        """Prepare PPO training batch from episode buffer with GAE - ONLY valid data."""
        episodes = buffer_state.episodes

        # Simple approach: reshape episodes and take first batch_size elements
        # This works because episode data is stored sequentially and we clear buffer after updates
        num_envs = episodes.obs.shape[1]
        obs_dim = episodes.obs.shape[2]
        action_dim = episodes.actions.shape[2]

        # Reshape episodes to flat form
        obs_flat = episodes.obs.reshape(-1, obs_dim)
        actions_flat = episodes.actions.reshape(-1, action_dim)
        log_probs_flat = episodes.log_probs.reshape(-1)
        rewards_flat = episodes.rewards.reshape(-1)
        values_flat = episodes.values.reshape(-1)
        dones_flat = episodes.dones.reshape(-1)

        # Take first batch_size elements (static slicing, no traced values)
        obs_batch = jax.lax.dynamic_slice(obs_flat, (0, 0), (batch_size, obs_dim))
        actions_batch = jax.lax.dynamic_slice(actions_flat, (0, 0), (batch_size, action_dim))
        log_probs_batch = jax.lax.dynamic_slice(log_probs_flat, (0,), (batch_size,))
        rewards_batch = jax.lax.dynamic_slice(rewards_flat, (0,), (batch_size,))
        values_batch = jax.lax.dynamic_slice(values_flat, (0,), (batch_size,))
        dones_batch = jax.lax.dynamic_slice(dones_flat, (0,), (batch_size,))

        # Compute GAE on the batch
        advantages_batch, returns_batch = self._compute_gae_advantages(
            rewards_batch, values_batch, dones_batch, num_envs
        )

        return PPOBatch(
            obs=obs_batch,
            actions=actions_batch,
            old_log_probs=log_probs_batch,
            advantages=advantages_batch,
            returns=returns_batch,
        )

    @partial(jax.jit, static_argnums=0)
    def update_step(self, state: PPOState, batch: PPOBatch, key: chex.PRNGKey) -> Tuple[PPOState, PPOInfo]:
        """Single PPO training step with clipped surrogate objective."""
        key_policy, key_value = jax.random.split(key, 2)

        # Update policy
        def policy_loss_fn(params: chex.ArrayTree) -> Tuple[chex.Array, PolicyInfo]:
            return self._policy_loss_fn(params, batch, key_policy)

        (policy_loss, policy_info), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(state.actor_params)

        policy_updates, new_actor_opt_state = self.actor_optimizer.update(policy_grads, state.actor_opt_state)
        new_actor_params = optax.apply_updates(state.actor_params, policy_updates)

        # Update value function
        def value_loss_fn(params: chex.ArrayTree) -> Tuple[chex.Array, ValueInfo]:
            return self._value_loss_fn(params, batch)

        (value_loss, value_info), value_grads = jax.value_and_grad(value_loss_fn, has_aux=True)(state.critic_params)

        value_updates, new_critic_opt_state = self.critic_optimizer.update(value_grads, state.critic_opt_state)
        new_critic_params = optax.apply_updates(state.critic_params, value_updates)

        # Create new state
        new_state = PPOState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            actor_opt_state=new_actor_opt_state,
            critic_opt_state=new_critic_opt_state,
        )

        # Combine info
        total_loss = policy_loss + self.config.value_loss_coef * value_loss
        info = PPOInfo(
            policy_info=policy_info,
            value_info=value_info,
            total_loss=total_loss,
        )

        return new_state, info

    @partial(jax.jit, static_argnums=0)
    def select_action_stochastic(self, state: PPOState, obs: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Select stochastic action given observation."""
        action, _ = self.actor.sample_action(state.actor_params, obs, key, training=False)
        return action

    @partial(jax.jit, static_argnums=0)
    def select_action_deterministic(self, state: PPOState, obs: chex.Array) -> chex.Array:
        """Select deterministic action given observation."""
        return self.actor.deterministic_action(state.actor_params, obs, training=False)

    @partial(jax.jit, static_argnums=0)
    def get_value(self, state: PPOState, obs: chex.Array) -> chex.Array:
        """Get value estimate for observations."""
        return self.critic.apply(state.critic_params, obs, training=False)

    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self, state: PPOState, obs: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Get action, log prob, and value for observations (used during rollout)."""
        action, log_prob = self.actor.sample_action(state.actor_params, obs, key, training=False)
        value = self.critic.apply(state.critic_params, obs, training=False)
        return action, log_prob, value

    def _policy_loss_fn(
        self,
        actor_params: chex.ArrayTree,
        batch: PPOBatch,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, PolicyInfo]:
        """Compute PPO policy loss with clipped surrogate objective."""
        # Get new action probabilities
        _, new_log_probs = self.actor.sample_action(actor_params, batch.obs, key, training=True)

        # Flatten log probs if needed
        new_log_probs = new_log_probs.squeeze(-1)
        old_log_probs = batch.old_log_probs.squeeze(-1) if batch.old_log_probs.ndim > 1 else batch.old_log_probs

        # Compute probability ratios
        ratio = jnp.exp(new_log_probs - old_log_probs)

        # Normalize advantages
        advantages = batch.advantages
        if self.config.normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Entropy bonus
        entropy = -jnp.mean(new_log_probs)
        entropy_loss = -self.config.entropy_coef * entropy

        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss

        # Compute metrics
        kl_divergence = jnp.mean(old_log_probs - new_log_probs)
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > self.config.clip_ratio).astype(DTYPE))

        policy_info = PolicyInfo(
            policy_loss=policy_loss,
            entropy=entropy,
            kl_divergence=kl_divergence,
            clip_fraction=clip_fraction,
        )

        return total_policy_loss, policy_info

    def _value_loss_fn(
        self,
        critic_params: chex.ArrayTree,
        batch: PPOBatch,
    ) -> Tuple[chex.Array, ValueInfo]:
        """Compute value function loss."""
        # Current value estimates
        values = self.critic.apply(critic_params, batch.obs, training=True).squeeze(-1)

        # Value loss (MSE)
        value_loss = jnp.mean((values - batch.returns) ** 2)

        # Compute explained variance
        value_mean = jnp.mean(values)
        total_variance = jnp.var(batch.returns)
        explained_variance = 1.0 - jnp.var(batch.returns - values) / (total_variance + 1e-8)

        value_info = ValueInfo(
            value_loss=value_loss,
            value_mean=value_mean,
            explained_variance=explained_variance,
        )

        return value_loss, value_info
