"""
Main script for Phase 1: JAX-based SAC for Cart-Pole Environment.
Demonstrates CPU-compatible implementation with all JAX APIs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import time
from tqdm import trange

# Import our modules
from environment.cartpole import CartPoleEnv
from algorithms.replay_buffer import ReplayBuffer
from algorithms.sac import SAC, SACConfig, SACState, Transition
from utils.visualization import CartPoleVisualizer, TrainingVisualizer


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
        key, action_key = jax.random.split(key)
        action = sac.select_action(sac_state, obs, action_key, deterministic=deterministic)

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
    """Main training loop for Phase 1 implementation."""
    print('🚀 Starting JAX-based SAC for Cart-Pole - Phase 1')
    print('=' * 50)

    # Configuration
    config = SACConfig(learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, hidden_dims=(8, 8))

    # Environment parameters
    num_envs = 4  # Start with small number for Phase 1
    max_episode_steps = 5  # TODO increase
    buffer_capacity = 100000
    batch_size = 256
    num_episodes = 100
    update_freq = 1
    eval_freq = 10

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
    cartpole_viz = CartPoleVisualizer(num_cartpoles=1, l=1.0, rail_limit=2.0, figsize=(8, 6))
    training_viz = TrainingVisualizer(figsize=(12, 6))

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
            episode_transitions = []

            # Collect episode data
            for step in trange(max_episode_steps, desc=f'Episode {episode}'):
                # Select action
                key, action_key = jax.random.split(key)
                action = sac.select_action(sac_state, obs, action_key, deterministic=False)

                # Take environment step
                next_obs, reward, done, next_env_state = env.step(env_state, action)

                # Store transitions for each environment
                obs_np = np.array(obs)
                next_obs_np = np.array(next_obs)
                reward_np = np.array(reward)
                done_np = np.array(done)
                action_np = np.array(action)

                for env_idx in range(num_envs):
                    transition = Transition(
                        obs=obs_np[env_idx],
                        action=action_np[env_idx],
                        reward=reward_np[env_idx],
                        next_obs=next_obs_np[env_idx],
                        done=done_np[env_idx],
                    )
                    episode_transitions.append(transition)

                episode_reward += float(jnp.mean(reward))

                # Update for next step
                obs = next_obs
                env_state = next_env_state

            # Add episode transitions to buffer
            for transition in episode_transitions:
                print(f'transition: {transition}')
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
                            actor_loss=float(info.actor_info.actor_loss),
                            critic_loss=float(info.critic_info.q1_loss),
                            alpha=float(info.alpha_info.alpha),
                            q_values=float(info.critic_info.q1_mean),
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

                # Update cart-pole visualization with current policy
                key, viz_key = jax.random.split(key)
                obs, env_state = env.reset()
                cartpole_viz.clear_trails()

                for _ in range(50):  # Show 50 steps of current policy
                    action = sac.select_action(sac_state, obs, viz_key, deterministic=True)
                    next_obs, _, _, next_env_state = env.step(env_state, action)

                    # Update cart-pole visualization with first environment's state
                    # obs format: [x, x_dot, cos(theta), sin(theta), theta_dot]
                    obs_first = np.array(obs)
                    cartpole_viz.update_cartpoles(obs_first)

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
    cartpole_viz.show(block=True)

    # Cleanup
    cartpole_viz.close()
    training_viz.close()

    print('✅ Phase 1 implementation complete!')


if __name__ == '__main__':
    main()
