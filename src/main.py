"""
Main script for JAX-based SAC for Cart-Pole Environment.
Update-based training with auto-reset environments.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from tqdm import trange

from environment.cartpole import CartPoleEnv
from algorithms.replay_buffer import ReplayBuffer, Transition
from algorithms.sac import SAC, AutoAlphaConfig, SACConfig
from utils.cartpole_viz import CartPoleLiveVisualizer
from utils.training_viz import TrainingVisualizer


def main():
    """Main training loop based on network updates."""
    print('🚀 Starting JAX-based SAC for Cart-Pole (Update-based Training)')
    print('=' * 60)

    # Configuration
    config = SACConfig(
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        hidden_dims=(128, 128),
    )

    # Training parameters
    num_envs = 256
    max_episode_steps = 1000
    total_updates = 200_000  # Total number of network updates to perform
    buffer_capacity = 1_000_000
    batch_size = 256
    updates_per_step = num_envs // 4  # Network updates per environment step
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

    # Create live visualizer
    if live_visualization:
        live_viz = CartPoleLiveVisualizer(num_cartpoles=min(num_envs, 4), length=env.length, rail_limit=env.rail_limit)
        print('🎮 Live pygame visualization enabled')
    else:
        live_viz = None

    print(f'Environment: {num_envs} cart-pole(s)')
    print(f'Network architecture: {config.hidden_dims}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Total updates: {total_updates}')
    print(f'Max episode steps: {max_episode_steps}')
    print(f'Batch size: {batch_size}')
    print('=' * 60)

    # Initialize environment and tracking
    obs, env_state = env.reset()
    env_steps = jnp.zeros(num_envs, dtype=jnp.int32)  # Track steps per environment
    episode_rewards = jnp.zeros(num_envs)  # Track cumulative reward per env
    total_episodes_completed = 0
    update_count = 0

    # Training metrics
    recent_rewards = []
    start_time = time.time()

    # Main training loop - continue until we've done all updates
    step = 0
    pbar = trange(total_updates, desc='Network Updates')

    try:
        while update_count < total_updates:
            # Select actions
            key, action_key = jax.random.split(key)
            action = sac.select_action_stochastic(sac_state, obs, action_key)

            # Take environment step
            next_obs, reward, done, next_env_state = env.step(env_state, action)

            # Update step counts and episode rewards
            env_steps += 1
            episode_rewards += reward

            # Check which environments should reset (done OR max steps reached)
            should_reset = done | (env_steps >= max_episode_steps)

            # Store completed episode rewards before reset
            completed_rewards = episode_rewards[should_reset]
            if jnp.any(should_reset):
                recent_rewards.extend(completed_rewards.tolist())
                total_episodes_completed += int(jnp.sum(should_reset))

                # Keep only recent rewards for visualization
                if len(recent_rewards) > 100:
                    recent_rewards = recent_rewards[-100:]

            # Store transitions in replay buffer
            buffer_state = replay_buffer.add_batch(
                buffer_state,
                Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done),
            )

            # Auto-reset environments that are done or reached max steps
            key, reset_key = jax.random.split(key)
            reset_obs, reset_env_state = env.reset()

            # Use reset values where needed, keep current values otherwise
            obs = jnp.where(should_reset[..., None], reset_obs, next_obs)
            env_state_dict = {}
            for field_name in next_env_state.__dataclass_fields__.keys():
                current_field = getattr(next_env_state, field_name)
                reset_field = getattr(reset_env_state, field_name)
                env_state_dict[field_name] = jnp.where(should_reset[..., None], reset_field, current_field)
            env_state = type(next_env_state)(**env_state_dict)

            # Reset tracking for environments that reset
            env_steps = jnp.where(should_reset, 0, env_steps)
            episode_rewards = jnp.where(should_reset, 0.0, episode_rewards)

            # Update live visualization
            if live_viz is not None:
                live_viz.update(env_state, episode=total_episodes_completed, step=step, rewards=np.array(reward))

            # Perform network updates if we have enough data
            if replay_buffer.can_sample(buffer_state, batch_size):
                for _ in range(updates_per_step):
                    if update_count >= total_updates:
                        break

                    key, sample_key = jax.random.split(key)
                    batch = replay_buffer.sample(buffer_state, sample_key, batch_size)

                    key, update_key = jax.random.split(key)
                    sac_state, info = sac.update_step(sac_state, batch, update_key)

                    update_count += 1
                    pbar.update(1)

                    # Update training visualization
                    if update_count % 10 == 0:
                        training_viz.update_metrics(
                            actor_loss=float(info.actor_info.actor_loss),
                            critic_loss=float(info.critic_info.q1_loss),
                            alpha=float(info.alpha_info.alpha),
                            q_values=float(info.critic_info.q1_mean),
                        )

                    # Update episode reward if we have recent data
                    if recent_rewards and update_count % 50 == 0:
                        avg_recent_reward = (
                            np.mean(recent_rewards[-20:]) if len(recent_rewards) >= 20 else np.mean(recent_rewards)
                        )
                        training_viz.update_metrics(episode_reward=avg_recent_reward)

            # Periodic logging and visualization updates
            if update_count % 1000 == 0 and update_count > 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
                updates_per_sec = update_count / elapsed_time if elapsed_time > 0 else 0

                avg_reward = (
                    np.mean(recent_rewards[-50:])
                    if len(recent_rewards) >= 50
                    else (np.mean(recent_rewards) if recent_rewards else 0.0)
                )

                # Update and show training plots
                training_viz.update_plots()
                training_viz.show(block=False)

                print(
                    f'Updates: {update_count:6d}/{total_updates} | '
                    f'Episodes: {total_episodes_completed:5d} | '
                    f'Avg Reward: {avg_reward:8.2f} | '
                    f'Buffer: {int(buffer_state.size):6d} | '
                    f'Steps/s: {steps_per_sec:.1f} | '
                    f'Updates/s: {updates_per_sec:.1f}'
                )

            step += 1

        pbar.close()

    except KeyboardInterrupt:
        print('\n⏸️  Training interrupted by user')

    # Final statistics
    print('\n' + '=' * 60)
    print('🏁 Training Complete')

    elapsed_time = time.time() - start_time
    print(f'Total updates completed: {update_count}')
    print(f'Total episodes completed: {total_episodes_completed}')
    print(f'Total environment steps: {step}')
    print(f'Training time: {elapsed_time:.1f}s')
    print(f'Average updates per second: {update_count / elapsed_time:.1f}')

    if recent_rewards:
        final_avg_reward = np.mean(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.mean(recent_rewards)
        print(f'Final average reward (last 50 episodes): {final_avg_reward:.2f}')

    # Final visualization update
    print('\n📊 Showing final visualizations...')
    training_viz.update_plots()
    training_viz.show(block=False)

    # Save training visualization
    training_viz.save('training_viz.png')
    print('📁 Training visualization saved as training_viz.png')

    # Cleanup
    training_viz.close()
    if live_viz is not None:
        live_viz.close()


if __name__ == '__main__':
    main()
