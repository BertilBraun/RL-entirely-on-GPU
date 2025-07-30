"""
Main script for JAX-based SAC for Cart-Pole Environment.
Demonstrates CPU-compatible implementation with all JAX APIs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import time
from tqdm import trange

from environment.cartpole import CartPoleEnv
from algorithms.replay_buffer import ReplayBuffer, Transition
from algorithms.sac import SAC, SACConfig, SACState
from utils.cartpole_viz import CartPoleLiveVisualizer
from utils.training_viz import TrainingVisualizer


def run_episode(
    env: CartPoleEnv,
    sac: SAC,
    sac_state: SACState,
    key: jax.Array,
    max_steps: int = 200,
    deterministic: bool = False,
) -> Tuple[float, int]:
    """Run a single episode and return total reward and steps."""
    obs, env_state = env.reset()

    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Select action
        if deterministic:
            action = sac.select_action_deterministic(sac_state, obs)
        else:
            key, action_key = jax.random.split(key)
            action = sac.select_action_stochastic(sac_state, obs, action_key)

        # Take environment step
        next_obs, reward, done, next_env_state = env.step(env_state, action)

        total_reward += float(jnp.mean(reward))  # Average reward across environments
        steps += 1

        # Update for next iteration
        obs = next_obs
        env_state = next_env_state

        # Check if episode should end (cart-pole envs typically don't terminate)
        if float(jnp.any(done)):
            break

    return total_reward, steps


def main():
    """Main training loop."""
    print('🚀 Starting JAX-based SAC for Cart-Pole')
    print('=' * 50)

    # Configuration
    config = SACConfig(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.5,
        auto_alpha=False,  # TODO true?
        hidden_dims=(32, 32),
    )

    # Environment parameters
    num_envs = 128
    max_episode_steps = 500  # TODO increase
    buffer_capacity = num_envs * max_episode_steps * 2  # approximately 2 episodes
    batch_size = 256
    num_episodes = 200
    update_freq = 1 / 8  # must be between 0 and 1
    eval_freq = 1  # TODO increase
    live_visualization = True

    # Initialize random key
    key = jax.random.PRNGKey(42)

    # Create environment
    env = CartPoleEnv(num_envs=num_envs)

    # Create SAC agent
    sac = SAC(obs_dim=env.obs_dim, action_dim=env.action_dim, max_action=env.max_force, config=config)

    # Initialize SAC state
    key, sac_key = jax.random.split(key)
    sac_state = sac.init_state(sac_key)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=env.obs_dim, action_dim=env.action_dim)

    # Initialize buffer state
    key, buffer_key = jax.random.split(key)
    buffer_state = replay_buffer.init_buffer_state(buffer_key)

    # Create visualizers
    training_viz = TrainingVisualizer(figsize=(12, 6))

    # Create live visualizer for real-time training visualization
    if live_visualization:
        live_viz = CartPoleLiveVisualizer(num_cartpoles=min(num_envs, 4), length=env.length, rail_limit=env.rail_limit)
        print('🎮 Live pygame visualization enabled')
    else:
        live_viz = None

    print(f'Environment: {num_envs} cart-pole(s)')
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
            obs, env_state = env.reset()

            episode_reward = 0.0

            # Clear trails for new episode
            if live_viz is not None:
                live_viz.clear_trails()

            # Collect episode data
            for step in trange(max_episode_steps, desc=f'Episode {episode}'):
                # Select action
                action = sac.select_action_deterministic(sac_state, obs)

                # Take environment step
                next_obs, reward, done, next_env_state = env.step(env_state, action)

                # Store transitions for each environment
                buffer_state = replay_buffer.add_batch(
                    buffer_state,
                    Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done),
                )

                # Update live visualization during training
                if live_viz is not None:
                    # Update visualization with current observations
                    live_viz.update(np.array(obs), episode=episode, step=step, rewards=np.array(reward))

                episode_reward += float(jnp.mean(reward))

                # Update for next step
                obs = next_obs
                env_state = next_env_state

                # Training updates
                if replay_buffer.can_sample(buffer_state, batch_size):
                    for _ in range(int(num_envs * update_freq)):
                        key, sample_key = jax.random.split(key)
                        batch = replay_buffer.sample(buffer_state, sample_key, batch_size)

                        key, update_key = jax.random.split(key)
                        sac_state, info = sac.update_step(sac_state, batch, update_key)

                        update_count += 1

                        # Update training visualization
                        if update_count % 10 == 0:
                            training_viz.update_metrics(
                                actor_loss=float(info.actor_info.actor_loss),
                                critic_loss=float(info.critic_info.q1_loss),
                                alpha=float(info.alpha_info.alpha),
                                q_values=float(info.critic_info.q1_mean),
                            )

            episode_rewards.append(episode_reward)

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

                # Update training plots
                training_viz.update_plots()
                training_viz.show(block=False)

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

    # save the training viz
    training_viz.save('training_viz.png')

    # Cleanup
    training_viz.close()
    if live_viz is not None:
        live_viz.close()


if __name__ == '__main__':
    main()
