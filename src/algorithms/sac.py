"""
Soft Actor-Critic (SAC) algorithm implementation using JAX.
"""

import jax
import jax.numpy as jnp
import chex
import optax
from typing import NamedTuple, Tuple

from networks.actor import ActorNetwork
from networks.critic import DoubleCriticNetwork
from algorithms.replay_buffer import Transition


class SACConfig(NamedTuple):
    """Configuration for SAC algorithm."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: float | None = None
    auto_alpha: bool = True
    hidden_dims: Tuple[int, ...] = (8,)


class SACState(NamedTuple):
    """State of the SAC algorithm."""

    actor_params: chex.ArrayTree
    critic_params: chex.ArrayTree
    target_critic_params: chex.ArrayTree
    alpha: chex.Array
    log_alpha: chex.Array

    actor_opt_state: chex.ArrayTree
    critic_opt_state: chex.ArrayTree
    alpha_opt_state: chex.ArrayTree | None = None


class SAC:
    """
    Soft Actor-Critic algorithm implementation.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_action: float, config: SACConfig = SACConfig()):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.config = config

        # Set target entropy if not provided
        if config.target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = config.target_entropy

        # Create networks
        self.actor = ActorNetwork(hidden_dims=config.hidden_dims, action_dim=action_dim, max_action=max_action)

        self.critic = DoubleCriticNetwork(hidden_dims=config.hidden_dims)

        # Create optimizers
        self.actor_optimizer = optax.adam(config.learning_rate)
        self.critic_optimizer = optax.adam(config.learning_rate)
        if config.auto_alpha:
            self.alpha_optimizer = optax.adam(config.learning_rate)
        else:
            self.alpha_optimizer = None

    def init_state(self, key: chex.PRNGKey) -> SACState:
        """Initialize SAC state with network parameters and optimizers."""
        key_actor, key_critic, key_alpha = jax.random.split(key, 3)

        # Initialize network parameters
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        actor_params = self.actor.init(key_actor, dummy_obs)
        critic_params = self.critic.init(key_critic, dummy_obs, dummy_action)
        target_critic_params = critic_params  # Initialize target as copy

        # Initialize alpha
        if self.config.auto_alpha:
            log_alpha = jnp.array(0.0)
            alpha = jnp.exp(log_alpha)
        else:
            alpha = jnp.array(self.config.alpha)
            log_alpha = jnp.log(alpha)

        # Initialize optimizers
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)
        if self.alpha_optimizer is not None:
            alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        else:
            alpha_opt_state = None

        return SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            alpha=alpha,
            log_alpha=log_alpha,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
        )

    @staticmethod
    @jax.jit
    def soft_update(target_params: chex.ArrayTree, params: chex.ArrayTree, tau: float) -> chex.ArrayTree:
        """Soft update of target network parameters."""
        return jax.tree.map(lambda t, p: (1 - tau) * t + tau * p, target_params, params)

    def critic_loss_fn(
        self,
        critic_params: chex.ArrayTree,
        target_critic_params: chex.ArrayTree,
        actor_params: chex.ArrayTree,
        batch: Transition,
        alpha: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, dict]:
        """Compute critic loss."""
        # Current Q-values
        q1_current, q2_current = self.critic.apply(critic_params, batch.obs, batch.action)

        # Target Q-values
        next_actions, next_log_probs = self.actor.sample_action(actor_params, batch.next_obs, key)

        q1_target, q2_target = self.critic.apply(target_critic_params, batch.next_obs, next_actions)

        # Take minimum of two target Q-values
        q_target = jnp.minimum(q1_target, q2_target)

        # Compute target with entropy regularization
        target_q = batch.reward + self.config.gamma * (1 - batch.done) * (q_target - alpha * next_log_probs)

        # Stop gradient on target
        target_q = jax.lax.stop_gradient(target_q)

        # Compute losses
        q1_loss = jnp.mean((q1_current - target_q) ** 2)
        q2_loss = jnp.mean((q2_current - target_q) ** 2)
        total_loss = q1_loss + q2_loss

        info = {
            'critic_loss': total_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'q1_mean': jnp.mean(q1_current),
            'q2_mean': jnp.mean(q2_current),
            'target_q_mean': jnp.mean(target_q),
        }

        return total_loss, info

    def actor_loss_fn(
        self,
        actor_params: chex.ArrayTree,
        critic_params: chex.ArrayTree,
        batch: Transition,
        alpha: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, dict]:
        """Compute actor loss."""
        # Sample actions
        actions, log_probs = self.actor.sample_action(actor_params, batch.obs, key)

        # Get Q-values for sampled actions
        q1, q2 = self.critic.apply(critic_params, batch.obs, actions)
        q_min = jnp.minimum(q1, q2)

        # Actor loss: maximize Q - alpha * log_pi
        actor_loss = jnp.mean(alpha * log_probs - q_min)

        info = {'actor_loss': actor_loss, 'log_probs_mean': jnp.mean(log_probs), 'q_actor_mean': jnp.mean(q_min)}

        return actor_loss, info

    def alpha_loss_fn(self, log_alpha: chex.Array, log_probs: chex.Array) -> Tuple[chex.Array, dict]:
        """Compute alpha (temperature) loss."""
        alpha = jnp.exp(log_alpha)
        alpha_loss = -jnp.mean(alpha * (log_probs + self.target_entropy))

        info = {'alpha_loss': alpha_loss, 'alpha': alpha}

        return alpha_loss, info

    def update_step(self, state: SACState, batch: Transition, key: chex.PRNGKey) -> Tuple[SACState, dict]:
        """Single training step."""
        key_critic, key_actor, key_alpha = jax.random.split(key, 3)

        # Update critic
        def critic_loss_fn(params: chex.ArrayTree) -> Tuple[chex.Array, dict]:
            return self.critic_loss_fn(
                params, state.target_critic_params, state.actor_params, batch, state.alpha, key_critic
            )

        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic_params)

        critic_updates, new_critic_opt_state = self.critic_optimizer.update(critic_grads, state.critic_opt_state)
        new_critic_params = optax.apply_updates(state.critic_params, critic_updates)

        # Update actor
        def actor_loss_fn(params: chex.ArrayTree) -> Tuple[chex.Array, dict]:
            return self.actor_loss_fn(params, new_critic_params, batch, state.alpha, key_actor)

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)

        actor_updates, new_actor_opt_state = self.actor_optimizer.update(actor_grads, state.actor_opt_state)
        new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

        # Update alpha (if auto)
        new_alpha = state.alpha
        new_log_alpha = state.log_alpha
        new_alpha_opt_state = state.alpha_opt_state
        alpha_info = {}

        if self.config.auto_alpha and self.alpha_optimizer is not None:
            # Get log probs for alpha update
            _, log_probs = self.actor.sample_action(new_actor_params, batch.obs, key_alpha)

            def alpha_loss_fn(log_alpha: chex.Array) -> Tuple[chex.Array, dict]:
                return self.alpha_loss_fn(log_alpha, log_probs)

            (alpha_loss, alpha_info), alpha_grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(state.log_alpha)

            alpha_updates, new_alpha_opt_state = self.alpha_optimizer.update(alpha_grads, state.alpha_opt_state)
            new_log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)
            new_alpha = jnp.exp(new_log_alpha)

        # Soft update target critic
        new_target_critic_params = self.soft_update(state.target_critic_params, new_critic_params, self.config.tau)

        # Create new state
        new_state = SACState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=new_target_critic_params,
            alpha=new_alpha,
            log_alpha=new_log_alpha,
            actor_opt_state=new_actor_opt_state,
            critic_opt_state=new_critic_opt_state,
            alpha_opt_state=new_alpha_opt_state,
        )

        # Combine info
        info = {**critic_info, **actor_info, **alpha_info}

        return new_state, info

    def select_action(
        self, state: SACState, obs: chex.Array, key: chex.PRNGKey, deterministic: bool = False
    ) -> chex.Array:
        """Select action given observation."""
        if deterministic:
            return self.actor.deterministic_action(state.actor_params, obs)
        else:
            action, _ = self.actor.sample_action(state.actor_params, obs, key)
            return action
