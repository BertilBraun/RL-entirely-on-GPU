"""
Chunk-based training logic for JAX-based RL algorithms.
"""

from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple
from abc import ABC, abstractmethod

from algorithms.base import Algorithm
from config import DTYPE
from environment.cartpole import CartPoleState
from environment.double_pendulum_cartpole import DoublePendulumCartPoleState
from training.data_structures import EnvType, TrainCarry, UpdateCarry, ChunkCarry, ChunkSummary, BufferStateType

EnvStateType = CartPoleState | DoublePendulumCartPoleState


class BaseChunkTrainer(ABC):
    """Base class for algorithm-specific chunk trainers."""

    def __init__(
        self,
        env: EnvType,
        algorithm: Algorithm,
        batch_size: int,
        updates_per_step: int,
        max_episode_steps: int,
        steps_per_gpu_chunk: int,
        ema_beta: float = 0.01,
        reward_ema_beta: float = 0.01,
    ) -> None:
        self.env = env
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.max_episode_steps = max_episode_steps
        self.steps_per_gpu_chunk = steps_per_gpu_chunk
        self.ema_beta = ema_beta
        self.reward_ema_beta = reward_ema_beta

    @partial(jax.jit, static_argnums=(0,))
    def run_chunk(self, train_carry: TrainCarry) -> Tuple[TrainCarry, ChunkSummary]:
        """Runs STEPS_PER_GPU_CHUNK env steps and updates on-device."""
        carry = ChunkCarry.init(train_carry)
        final_carry, _ = jax.lax.scan(self._one_step, carry, xs=None, length=self.steps_per_gpu_chunk)
        summary = ChunkSummary.from_carry(final_carry)
        return final_carry.train, summary

    def _one_step(self, carry: ChunkCarry, _) -> Tuple[ChunkCarry, None]:
        """Execute one environment step with updates."""
        # Split RNG for different operations
        rng, action_key, reset_key = jax.random.split(carry.train.rng, 3)

        # Algorithm-specific rollout data collection
        action, rollout_data = self.algorithm.collect_rollout_data(
            carry.train.algorithm_state, carry.train.obs, action_key
        )

        next_obs, reward, done, next_env_state = self.env.step(carry.train.env_state, action)

        # Handle episode termination and reset
        env_state_after_reset, obs_after_reset, env_steps_after_reset, rewards_after_reset = self._handle_episode_reset(
            carry, next_obs, next_env_state, reward, done, reset_key
        )

        # Algorithm-specific buffer update
        buffer_state_updated = self.algorithm.update_buffer(
            carry.train.buffer_state,
            rollout_data,
            carry.train.obs,
            action,
            reward,
            next_obs,
            done,
        )

        # Parameter updates
        updated_carry = self._perform_parameter_updates(
            rng,
            carry,
            buffer_state_updated,
            env_state_after_reset,
            obs_after_reset,
            env_steps_after_reset,
            rewards_after_reset,
            reward.astype(DTYPE),
        )

        return updated_carry, None

    def _handle_episode_reset(
        self,
        carry: ChunkCarry,
        next_obs: chex.Array,
        next_env_state: EnvStateType,
        reward: chex.Array,
        done: chex.Array,
        reset_key: chex.PRNGKey,
    ) -> Tuple[EnvStateType, chex.Array, chex.Array, chex.Array]:
        """Handle episode termination and environment reset."""
        reset_obs, reset_state = self.env.reset(reset_key)

        next_env_steps = carry.train.env_steps + 1
        next_episode_rewards = carry.train.episode_rewards + reward.astype(DTYPE)
        should_reset = done | (next_env_steps >= self.max_episode_steps)

        # NOTE: for some reason, [..., None] is needed just here
        obs_after_reset = jnp.where(should_reset[..., None], reset_obs, next_obs)

        # Handle state reset based on environment type
        if isinstance(next_env_state, DoublePendulumCartPoleState):
            env_state_after_reset = DoublePendulumCartPoleState(
                x=jnp.where(should_reset, reset_state.x, next_env_state.x),
                x_dot=jnp.where(should_reset, reset_state.x_dot, next_env_state.x_dot),
                theta1=jnp.where(should_reset, reset_state.theta1, next_env_state.theta1),
                theta1_dot=jnp.where(should_reset, reset_state.theta1_dot, next_env_state.theta1_dot),
                theta2=jnp.where(should_reset, reset_state.theta2, next_env_state.theta2),
                theta2_dot=jnp.where(should_reset, reset_state.theta2_dot, next_env_state.theta2_dot),
            )
        else:
            env_state_after_reset = CartPoleState(
                x=jnp.where(should_reset, reset_state.x, next_env_state.x),
                x_dot=jnp.where(should_reset, reset_state.x_dot, next_env_state.x_dot),
                theta=jnp.where(should_reset, reset_state.theta, next_env_state.theta),
                theta_dot=jnp.where(should_reset, reset_state.theta_dot, next_env_state.theta_dot),
            )
        env_steps_after_reset = jnp.where(should_reset, 0, next_env_steps)
        rewards_after_reset = jnp.where(should_reset, 0.0, next_episode_rewards)

        return env_state_after_reset, obs_after_reset, env_steps_after_reset, rewards_after_reset

    def _perform_parameter_updates(
        self,
        rng: chex.PRNGKey,
        carry: ChunkCarry,
        buffer_state: BufferStateType,
        env_state: EnvStateType,
        obs: chex.Array,
        env_steps: chex.Array,
        episode_rewards: chex.Array,
        step_reward: chex.Array,
    ) -> ChunkCarry:
        """Perform algorithm parameter updates and update EMAs."""
        # Initialize update carry
        uc0 = UpdateCarry.init(carry, rng, buffer_state)

        # Run parameter updates
        uc_final, _ = jax.lax.scan(self._single_update, uc0, xs=None, length=self.updates_per_step)

        # Update reward EMA
        step_reward_mean = jnp.mean(step_reward)
        reward_ema_updated = (1 - self.reward_ema_beta) * carry.reward_ema + self.reward_ema_beta * step_reward_mean

        # Create updated carry
        return ChunkCarry(
            train=TrainCarry(
                rng=uc_final.rng,
                algorithm_state=uc_final.algorithm_state,
                buffer_state=uc_final.buffer_state,
                env_state=env_state,
                obs=obs,
                env_steps=env_steps,
                episode_rewards=episode_rewards,
                total_updates_done=uc_final.total_updates_done,
            ),
            chunk_updates_done=uc_final.chunk_updates_done,
            actor_loss_ema=uc_final.actor_loss_ema,
            critic_loss_ema=uc_final.critic_loss_ema,
            alpha_ema=uc_final.alpha_ema,
            q_ema=uc_final.q_ema,
            reward_ema=reward_ema_updated,
        )

    @abstractmethod
    def _single_update(self, ucc: UpdateCarry, _) -> Tuple[UpdateCarry, None]:
        """Perform a single algorithm parameter update (algorithm-specific)."""
        pass


class SACChunkTrainer(BaseChunkTrainer):
    """SAC-specific chunk trainer."""

    def _single_update(self, ucc: UpdateCarry, _) -> Tuple[UpdateCarry, None]:
        """Perform a single SAC parameter update."""
        next_rng, sample_key, update_key = jax.random.split(ucc.rng, 3)

        # SAC batch preparation and update
        batch = self.algorithm.prepare_update_batch(ucc.buffer_state, sample_key, self.batch_size)
        next_algorithm_state, info = self.algorithm.update_step(ucc.algorithm_state, batch, update_key)

        # Extract SAC-specific metrics
        actor_loss = info.actor_info.actor_loss
        critic_loss = info.critic_info.q1_loss
        alpha = info.alpha_info.alpha
        q_mean = info.critic_info.q1_mean

        beta = jnp.asarray(self.ema_beta, dtype=DTYPE)
        return UpdateCarry(
            rng=next_rng,
            algorithm_state=next_algorithm_state,
            buffer_state=ucc.buffer_state,
            total_updates_done=ucc.total_updates_done + 1,
            chunk_updates_done=ucc.chunk_updates_done + 1,
            actor_loss_ema=(1 - beta) * ucc.actor_loss_ema + beta * actor_loss,
            critic_loss_ema=(1 - beta) * ucc.critic_loss_ema + beta * critic_loss,
            alpha_ema=(1 - beta) * ucc.alpha_ema + beta * alpha,
            q_ema=(1 - beta) * ucc.q_ema + beta * q_mean,
        ), None


class PPOChunkTrainer(BaseChunkTrainer):
    """PPO-specific chunk trainer."""

    def _perform_parameter_updates(
        self,
        rng: chex.PRNGKey,
        carry: ChunkCarry,
        buffer_state: BufferStateType,
        env_state: EnvStateType,
        obs: chex.Array,
        env_steps: chex.Array,
        episode_rewards: chex.Array,
        step_reward: chex.Array,
    ) -> ChunkCarry:
        """Perform PPO parameter updates with proper epochs and buffer management."""
        # For PPO, we want to do multiple epochs on the same data before clearing
        # So we override the base implementation

        # Initialize update carry
        uc0 = UpdateCarry.init(carry, rng, buffer_state)

        # Do all PPO epochs on the same data
        uc_final, _ = jax.lax.scan(self._single_update_no_clear, uc0, xs=None, length=self.updates_per_step)

        # Clear buffer ONLY after all epochs are complete
        from algorithms.episode_buffer import EpisodeBuffer

        cleared_buffer_state = EpisodeBuffer.clear_buffer(uc_final.buffer_state)

        # Update reward EMA
        step_reward_mean = jnp.mean(step_reward)
        reward_ema_updated = (1 - self.reward_ema_beta) * carry.reward_ema + self.reward_ema_beta * step_reward_mean

        # Create updated carry with cleared buffer
        return ChunkCarry(
            train=TrainCarry(
                rng=uc_final.rng,
                algorithm_state=uc_final.algorithm_state,
                buffer_state=cleared_buffer_state,  # Use cleared buffer
                env_state=env_state,
                obs=obs,
                env_steps=env_steps,
                episode_rewards=episode_rewards,
                total_updates_done=uc_final.total_updates_done,
            ),
            chunk_updates_done=uc_final.chunk_updates_done,
            actor_loss_ema=uc_final.actor_loss_ema,
            critic_loss_ema=uc_final.critic_loss_ema,
            alpha_ema=uc_final.alpha_ema,
            q_ema=uc_final.q_ema,
            reward_ema=reward_ema_updated,
        )

    def _single_update_no_clear(self, ucc: UpdateCarry, _) -> Tuple[UpdateCarry, None]:
        """Perform a single PPO parameter update WITHOUT clearing buffer."""
        next_rng, sample_key, update_key = jax.random.split(ucc.rng, 3)

        # PPO batch preparation and update
        batch = self.algorithm.prepare_update_batch(ucc.buffer_state, sample_key, self.batch_size)
        next_algorithm_state, info = self.algorithm.update_step(ucc.algorithm_state, batch, update_key)

        # DON'T clear buffer - keep using the same data for multiple epochs

        # Extract PPO-specific metrics
        actor_loss = info.policy_info.policy_loss
        critic_loss = info.value_info.value_loss
        alpha = jnp.array(0.0, dtype=DTYPE)  # PPO doesn't use alpha
        q_mean = info.value_info.value_mean

        beta = jnp.asarray(self.ema_beta, dtype=DTYPE)
        return UpdateCarry(
            rng=next_rng,
            algorithm_state=next_algorithm_state,
            buffer_state=ucc.buffer_state,  # Keep same buffer for next epoch
            total_updates_done=ucc.total_updates_done + 1,
            chunk_updates_done=ucc.chunk_updates_done + 1,
            actor_loss_ema=(1 - beta) * ucc.actor_loss_ema + beta * actor_loss,
            critic_loss_ema=(1 - beta) * ucc.critic_loss_ema + beta * critic_loss,
            alpha_ema=(1 - beta) * ucc.alpha_ema + beta * alpha,
            q_ema=(1 - beta) * ucc.q_ema + beta * q_mean,
        ), None

    def _single_update(self, ucc: UpdateCarry, _) -> Tuple[UpdateCarry, None]:
        """This method is not used for PPO - we use _single_update_no_clear instead."""
        # This should never be called for PPO, but just in case
        return self._single_update_no_clear(ucc, _)


def create_chunk_trainer(
    env: EnvType,
    algorithm: Algorithm,
    batch_size: int,
    updates_per_step: int,
    max_episode_steps: int,
    steps_per_gpu_chunk: int,
    ema_beta: float = 0.01,
    reward_ema_beta: float = 0.01,
) -> BaseChunkTrainer:
    """Factory function to create the appropriate chunk trainer for the algorithm."""
    if algorithm.algorithm_name == 'SAC':
        return SACChunkTrainer(
            env,
            algorithm,
            batch_size,
            updates_per_step,
            max_episode_steps,
            steps_per_gpu_chunk,
            ema_beta,
            reward_ema_beta,
        )
    elif algorithm.algorithm_name == 'PPO':
        return PPOChunkTrainer(
            env,
            algorithm,
            batch_size,
            updates_per_step,
            max_episode_steps,
            steps_per_gpu_chunk,
            ema_beta,
            reward_ema_beta,
        )
    else:
        raise NotImplementedError(f'No chunk trainer implemented for {algorithm.algorithm_name}')


# Legacy alias for backwards compatibility
ChunkTrainer = BaseChunkTrainer
