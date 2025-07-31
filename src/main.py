# src/main.py
"""
Main script for JAX-based SAC for Cart-Pole Environment.
Chunked training: run N updates entirely on GPU, then log/viz on host, repeat.
"""

from __future__ import annotations

import time
import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

from config import (
    SAC_CONFIG,
    NUM_ENVS,
    MAX_EPISODE_STEPS,
    TOTAL_UPDATES,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    UPDATES_PER_STEP,
    STEPS_PER_GPU_CHUNK,
    EMA_BETA,
    ENABLE_TRAINING_VIZ,
    ENABLE_LIVE_VIZ,
    USE_DOUBLE_PENDULUM,
)
from environment.cartpole import CartPoleEnv
from environment.double_pendulum_cartpole import DoublePendulumCartPoleEnv
from algorithms.replay_buffer import ReplayBuffer
from algorithms.sac import SAC, SACState
from training.data_structures import EnvType, TrainCarry, TrainingSetup
from training.chunk_trainer import ChunkTrainer
from utils.cartpole_viz import CartPoleLiveVisualizer
from utils.double_pendulum_cartpole_viz import DoublePendulumCartPoleLiveVisualizer
from utils.training_viz import TrainingVisualizer


VisualizerType = DoublePendulumCartPoleLiveVisualizer | CartPoleLiveVisualizer


def setup_environment_and_agent(rng: jax.Array) -> TrainingSetup:
    """Initialize environment, agent, and replay buffer."""
    env: EnvType
    if USE_DOUBLE_PENDULUM:
        env = DoublePendulumCartPoleEnv(num_envs=NUM_ENVS)
    else:
        env = CartPoleEnv(num_envs=NUM_ENVS)
    sac = SAC(obs_dim=env.obs_dim, action_dim=env.action_dim, max_action=env.max_force, config=SAC_CONFIG)

    rng, sac_key, buf_key, reset_key = jax.random.split(rng, 4)
    sac_state = sac.init_state(sac_key)
    sac_state = sac_state.try_load()

    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, obs_dim=env.obs_dim, action_dim=env.action_dim)
    buffer_state = replay_buffer.init_buffer_state(buf_key)

    # Initial obs/state via functional reset
    obs0, env_state0 = env.reset(reset_key)

    return TrainingSetup(
        env=env,
        sac=sac,
        sac_state=sac_state,
        buffer_state=buffer_state,
        initial_obs=obs0,
        initial_env_state=env_state0,
        rng=rng,
    )


def setup_visualizers(env: EnvType) -> Tuple[Optional[TrainingVisualizer], Optional[VisualizerType]]:
    """Initialize visualization components."""
    training_viz = TrainingVisualizer(figsize=(12, 6)) if ENABLE_TRAINING_VIZ else None

    live_viz: VisualizerType | None
    if ENABLE_LIVE_VIZ:
        if USE_DOUBLE_PENDULUM:
            assert isinstance(env, DoublePendulumCartPoleEnv)
            live_viz = DoublePendulumCartPoleLiveVisualizer(
                num_cartpoles=min(NUM_ENVS, 4),
                length1=env.params.length1,
                length2=env.params.length2,
                rail_limit=env.rail_limit,
            )
        else:
            assert isinstance(env, CartPoleEnv)
            live_viz = CartPoleLiveVisualizer(
                num_cartpoles=min(NUM_ENVS, 4), length=env.length, rail_limit=env.rail_limit
            )
    else:
        live_viz = None

    return training_viz, live_viz


def print_training_info() -> None:
    """Print training configuration."""
    env_type = 'Double Pendulum Cart-Pole' if USE_DOUBLE_PENDULUM else 'Cart-Pole'
    print(f'🚀 Starting JAX-based SAC for {env_type} (Chunked GPU Training)')
    print('=' * 80)
    print(f'Environment: {NUM_ENVS} {env_type.lower()}(s)')
    print(f'Network: {SAC_CONFIG.hidden_dims} | LR: {SAC_CONFIG.learning_rate}')
    print(f'Updates: total={TOTAL_UPDATES}, per-step={UPDATES_PER_STEP}, per-chunk={STEPS_PER_GPU_CHUNK}')
    print(f'Max episode steps: {MAX_EPISODE_STEPS} | Batch size: {BATCH_SIZE}')
    print('=' * 80)


def run_training_loop(
    train_carry: TrainCarry,
    chunk_trainer: ChunkTrainer,
    training_viz: Optional[TrainingVisualizer],
    live_viz: Optional[VisualizerType],
) -> float:
    """Main training loop."""
    start_time = time.time()

    try:
        while int(train_carry.total_updates_done) < TOTAL_UPDATES:
            t0 = time.time()
            train_carry, metrics = chunk_trainer.run_chunk(train_carry)  # GPU-only work

            # Access so that the results of the JAX GPU computation must be materialized before the dt is calculated correctly
            int(metrics.chunk_updates)
            dt = time.time() - t0

            # update visualizations
            if training_viz:
                training_viz.update_metrics(
                    actor_loss=metrics.actor_loss,
                    critic_loss=metrics.critic_loss,
                    alpha=metrics.alpha,
                    q_values=metrics.q_values,
                    episode_reward=metrics.reward,
                )
                training_viz.update_plots()
                training_viz.show(block=False)

            if live_viz:
                live_viz.update(
                    train_carry.env_state,  # type: ignore
                    step=int(train_carry.total_updates_done),
                    rewards=np.array([train_carry.episode_rewards]).squeeze(),
                )

            # Print progress
            ups = metrics.chunk_updates / dt if dt > 0 else 0.0
            print(
                f'+{metrics.chunk_updates:4d} updates this chunk | {ups:6.1f} upd/s | '
                f'total {int(train_carry.total_updates_done):6d}/{TOTAL_UPDATES} | '
                f'actor {metrics.actor_loss:.3f} | critic {metrics.critic_loss:.3f} | '
                f'alpha {metrics.alpha:.3f} | q {metrics.q_values:.2f} | r {metrics.reward:.2f}'
            )

            train_carry.sac_state.save_model(int(train_carry.total_updates_done))

    except KeyboardInterrupt:
        print('\n⏸️  Training interrupted by user')

    run_pendulums_viz(train_carry.rng, chunk_trainer.sac, train_carry.sac_state)

    return time.time() - start_time


def print_final_stats_and_cleanup(
    train_carry: TrainCarry,
    elapsed_time: float,
    training_viz: Optional[TrainingVisualizer],
    live_viz: Optional[VisualizerType],
) -> None:
    """Print final statistics and clean up visualizations."""
    total_updates_done = int(train_carry.total_updates_done)

    print('\n' + '=' * 70)
    print('🏁 Training Complete')
    print(f'Total updates completed: {total_updates_done}')
    print(f'Training time: {elapsed_time:.1f}s')
    if elapsed_time > 0:
        print(f'Average updates per second: {total_updates_done / elapsed_time:.1f}')

    if training_viz:
        print('\n📊 Saving final visualizations...')
        training_viz.update_plots()
        training_viz.save('training_viz.png')
        print('📁 Training visualization saved as training_viz.png')
        training_viz.close()
    if live_viz:
        live_viz.close()


def run_pendulums_viz(rng: chex.PRNGKey, sac: SAC, sac_state: SACState) -> None:
    """Run visualization of trained agent."""
    if USE_DOUBLE_PENDULUM:
        env = DoublePendulumCartPoleEnv(num_envs=4)
        live_viz = DoublePendulumCartPoleLiveVisualizer(
            num_cartpoles=env.num_envs,
            length1=env.params.length1,
            length2=env.params.length2,
            rail_limit=env.rail_limit,
            should_save=True,
        )
        filename = 'double_pendulums_viz.gif'
    else:
        env = CartPoleEnv(num_envs=4)
        live_viz = CartPoleLiveVisualizer(
            num_cartpoles=env.num_envs, length=env.length, rail_limit=env.rail_limit, should_save=True
        )
        filename = 'pendulums_viz.gif'

    obs, env_state = env.reset(rng)

    # run until all pendulums are done
    for step in range(1000):
        action = sac.select_action_deterministic(sac_state, obs)
        obs, reward, done, env_state = env.step(env_state, action)
        live_viz.update(env_state, step=step, rewards=np.array([reward]).squeeze())
        if jnp.any(done):
            break

    live_viz.save_frames(filename)
    print(f'📁 Visualization saved as {filename}')


def main() -> None:
    """Main training function."""
    print_training_info()

    # RNG
    rng = jax.random.PRNGKey(42)

    # Setup components
    setup = setup_environment_and_agent(rng)
    training_viz, live_viz = setup_visualizers(setup.env)

    # Initialize training state
    train_carry = TrainCarry.init(setup)

    # Create chunk trainer
    chunk_trainer = ChunkTrainer(
        env=setup.env,
        sac=setup.sac,
        batch_size=BATCH_SIZE,
        updates_per_step=UPDATES_PER_STEP,
        max_episode_steps=MAX_EPISODE_STEPS,
        steps_per_gpu_chunk=STEPS_PER_GPU_CHUNK,
        ema_beta=EMA_BETA,
    )

    # Run training
    elapsed_time = run_training_loop(train_carry, chunk_trainer, training_viz, live_viz)

    # Final cleanup
    print_final_stats_and_cleanup(train_carry, elapsed_time, training_viz, live_viz)


if __name__ == '__main__':
    main()
