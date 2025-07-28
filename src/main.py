"""
Main script for Phase 1: JAX-based SAC for Pendulum Environment.
Demonstrates CPU-compatible implementation with all JAX APIs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import time
import chex

# Import our modules
from environment.pendulum import PendulumEnv
from algorithms.replay_buffer import ReplayBuffer
from algorithms.sac import SAC, SACConfig, SACState, Transition
from utils.visualization import PendulumVisualizer, TrainingVisualizer


def create_transition(
    obs: chex.Array, action: chex.Array, reward: chex.Array, next_obs: chex.Array, done: chex.Array
) -> Transition:
    """Helper function to create a Transition."""
    return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)


def run_episode(
    env: PendulumEnv,
    sac: SAC,
    sac_state: SACState,
    key: jax.random.PRNGKey,
    max_steps: int = 200,
    deterministic: bool = False,
) -> Tuple[float, int]:
    """Run a single episode and return total reward and steps."""
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key)

    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Select action
        key, action_key = jax.random.split(key)
        action = sac.select_action(sac_state, obs, action_key, deterministic=deterministic)

        # Take environment step
        next_obs, reward, done, next_env_state = env.step(env_state, action)

        total_reward += float(reward)
        steps += 1

        # Update for next iteration
        obs = next_obs
        env_state = next_env_state

        # Check if episode should end (pendulum envs typically don't terminate)
        if float(done):
            break

    return total_reward, steps


def main():
    """Main training loop for Phase 1 implementation."""
    print('🚀 Starting JAX-based SAC for Pendulum - Phase 1')
    print('=' * 50)

    # Configuration
    config = SACConfig(learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, hidden_dims=(8, 8))

    # Environment parameters
    num_envs = 4  # Start with small number for Phase 1
    max_episode_steps = 200
    buffer_capacity = 100000
    batch_size = 256
    num_episodes = 100
    update_freq = 1
    eval_freq = 10

    # Initialize random key
    key = jax.random.PRNGKey(42)

    # Create environment
    env = PendulumEnv(
        num_envs=1,  # Single environment for Phase 1
        max_torque=2.0,
        dt=0.05,
    )

    # Create SAC agent
    sac = SAC(obs_dim=3, action_dim=1, max_action=2.0, config=config)

    # Initialize SAC state
    key, sac_key = jax.random.split(key)
    sac_state = sac.init_state(sac_key)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=3, action_dim=1)

    # Initialize buffer state
    key, buffer_key = jax.random.split(key)
    buffer_state = replay_buffer.init_buffer_state(buffer_key)

    # Create visualizers
    pendulum_viz = PendulumVisualizer(num_pendulums=1, figsize=(6, 6))
    training_viz = TrainingVisualizer(figsize=(12, 6))

    print(f'Environment: {num_envs} pendulum(s)')
    print(f'Network architecture: {config.hidden_dims}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Replay buffer capacity: {buffer_capacity}')
    print(f'Batch size: {batch_size}')
    print('=' * 50)

    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    update_count = 0

    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()

            # Run episode
            key, episode_key = jax.random.split(key)
            obs, env_state = env.reset(episode_key)

            episode_reward = 0.0
            episode_transitions = []

            # Collect episode data
            for step in range(max_episode_steps):
                # Select action
                key, action_key = jax.random.split(key)
                action = sac.select_action(sac_state, obs, action_key, deterministic=False)

                # Take environment step
                next_obs, reward, done, next_env_state = env.step(env_state, action)

                # Store transition
                transition = create_transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
                episode_transitions.append(transition)

                episode_reward += float(reward)

                # Update for next step
                obs = next_obs
                env_state = next_env_state

            # Add episode transitions to buffer
            for transition in episode_transitions:
                buffer_state = replay_buffer.add(buffer_state, transition)

            episode_rewards.append(episode_reward)

            # Training updates
            if replay_buffer.can_sample(buffer_state, batch_size):
                for _ in range(len(episode_transitions) // update_freq):
                    key, sample_key = jax.random.split(key)
                    batch = replay_buffer.sample(buffer_state, sample_key, batch_size)

                    key, update_key = jax.random.split(key)
                    sac_state, info = sac.update_step(sac_state, batch, update_key)

                    update_count += 1

                    # Update training visualization
                    if update_count % 10 == 0:
                        training_viz.update_metrics(
                            actor_loss=float(info.get('actor_loss', 0)),
                            critic_loss=float(info.get('critic_loss', 0)),
                            alpha=float(info.get('alpha', config.alpha)),
                            q_values=float(info.get('q1_mean', 0)),
                        )

            # Update visualizations
            training_viz.update_metrics(episode_reward=episode_reward)

            # Evaluation
            if episode % eval_freq == 0:
                key, eval_key = jax.random.split(key)
                eval_reward, eval_steps = run_episode(
                    env, sac, sac_state, eval_key, max_steps=max_episode_steps, deterministic=True
                )

                if eval_reward > best_reward:
                    best_reward = eval_reward

                # Update pendulum visualization with current policy
                key, viz_key = jax.random.split(key)
                obs, env_state = env.reset(viz_key)
                pendulum_viz.clear_trails()

                for _ in range(50):  # Show 50 steps of current policy
                    action = sac.select_action(sac_state, obs, viz_key, deterministic=True)
                    next_obs, _, _, next_env_state = env.step(env_state, action)

                    # Update pendulum visualization
                    theta = jnp.arctan2(obs[1], obs[0])  # Extract theta from [cos, sin, theta_dot]
                    pendulum_viz.update_pendulums(jnp.array([theta]))

                    obs = next_obs
                    env_state = next_env_state

                # Update training plots
                training_viz.update_plots()

                print(
                    f'Episode {episode:4d} | '
                    f'Reward: {episode_reward:8.2f} | '
                    f'Eval: {eval_reward:8.2f} | '
                    f'Best: {best_reward:8.2f} | '
                    f'Buffer: {int(buffer_state.size):6d} | '
                    f'Updates: {update_count:6d} | '
                    f'Time: {time.time() - episode_start_time:.2f}s'
                )

    except KeyboardInterrupt:
        print('\n⏸️  Training interrupted by user')

    # Final evaluation
    print('\n' + '=' * 50)
    print('🏁 Final Evaluation')

    final_rewards = []
    for i in range(10):
        key, eval_key = jax.random.split(key)
        eval_reward, _ = run_episode(env, sac, sac_state, eval_key, max_steps=max_episode_steps, deterministic=True)
        final_rewards.append(eval_reward)

    avg_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)

    print(f'Average reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Best episode reward: {best_reward:.2f}')
    print(f'Total training episodes: {len(episode_rewards)}')
    print(f'Total updates: {update_count}')

    # Show visualizations
    print('\n📊 Showing visualizations...')
    training_viz.update_plots()
    training_viz.show(block=False)
    pendulum_viz.show(block=True)

    # Cleanup
    pendulum_viz.close()
    training_viz.close()

    print('✅ Phase 1 implementation complete!')


if __name__ == '__main__':
    main()
