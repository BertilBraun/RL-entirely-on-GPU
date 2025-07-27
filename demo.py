"""
Simple demo script to test Phase 1 JAX-based SAC implementation.
This script runs a quick test to verify all components work correctly.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import numpy as np


def test_pendulum_environment():
    """Test the pendulum environment."""
    print('🧪 Testing Pendulum Environment...')

    from src.environment.pendulum import PendulumEnv, pendulum_step, get_obs, reward_fn

    # Test individual functions
    theta = jnp.array(0.5)
    theta_dot = jnp.array(0.1)
    torque = jnp.array(1.0)

    # Test physics step
    next_theta, next_theta_dot = pendulum_step(theta, theta_dot, torque)
    print(f'  Physics step: θ={float(next_theta):.3f}, θ̇={float(next_theta_dot):.3f}')

    # Test observation
    obs = get_obs(theta, theta_dot)
    print(f'  Observation: {obs}')

    # Test reward
    reward = reward_fn(theta, theta_dot, torque)
    print(f'  Reward: {float(reward):.3f}')

    # Test environment class
    env = PendulumEnv(num_envs=1)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    action = jnp.array([0.5])
    next_obs, reward, done, next_state = env.step(state, action)

    print(f'  Environment step successful!')
    print('✅ Pendulum Environment test passed!')


def test_networks():
    """Test the neural networks."""
    print('🧪 Testing Neural Networks...')

    from src.networks.actor import ActorNetwork
    from src.networks.critic import DoubleCriticNetwork

    # Test actor network
    actor = ActorNetwork(hidden_dims=(64, 64), action_dim=1, max_action=2.0)
    key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, 3))

    actor_params = actor.init(key, dummy_obs)
    mu, log_std = actor.apply(actor_params, dummy_obs)
    print(f'  Actor output shapes: mu={mu.shape}, log_std={log_std.shape}')

    # Test critic network
    critic = DoubleCriticNetwork(hidden_dims=(64, 64))
    dummy_action = jnp.zeros((1, 1))

    critic_params = critic.init(key, dummy_obs, dummy_action)
    q1, q2 = critic.apply(critic_params, dummy_obs, dummy_action)
    print(f'  Critic output shapes: q1={q1.shape}, q2={q2.shape}')

    print('✅ Neural Networks test passed!')


def test_replay_buffer():
    """Test the replay buffer."""
    print('🧪 Testing Replay Buffer...')

    from src.algorithms.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=1000, obs_dim=3, action_dim=1)
    key = jax.random.PRNGKey(42)
    buffer_state = buffer.init_buffer_state(key)

    # Test adding transitions
    transition = Transition(
        obs=jnp.array([1.0, 0.0, 0.5]),
        action=jnp.array([0.1]),
        reward=jnp.array(-1.0),
        next_obs=jnp.array([0.9, 0.1, 0.4]),
        done=jnp.array(False),
    )

    buffer_state = buffer.add(buffer_state, transition)
    print(f'  Buffer size after adding: {buffer_state.size}')

    # Add more transitions
    for i in range(10):
        key, subkey = jax.random.split(key)
        random_transition = Transition(
            obs=jax.random.normal(subkey, (3,)),
            action=jax.random.normal(subkey, (1,)),
            reward=jax.random.normal(subkey, ()),
            next_obs=jax.random.normal(subkey, (3,)),
            done=jnp.array(False),
        )
        buffer_state = buffer.add(buffer_state, random_transition)

    print(f'  Buffer size after adding more: {buffer_state.size}')

    # Test sampling
    key, sample_key = jax.random.split(key)
    batch = buffer.sample(buffer_state, sample_key, batch_size=5)
    print(f'  Sampled batch shapes: obs={batch.obs.shape}, action={batch.action.shape}')

    print('✅ Replay Buffer test passed!')


def test_simple_training():
    """Test a very simple training loop."""
    print('🧪 Testing Simple Training Loop...')

    from src.environment.pendulum import PendulumEnv
    from src.algorithms.sac import SAC, SACConfig
    from src.algorithms.replay_buffer import ReplayBuffer, Transition

    # Create environment and agent
    env = PendulumEnv(num_envs=1)
    config = SACConfig(hidden_dims=(32,))
    sac = SAC(obs_dim=3, action_dim=1, max_action=2.0, config=config)

    # Initialize
    key = jax.random.PRNGKey(42)
    key, sac_key = jax.random.split(key)
    sac_state = sac.init_state(sac_key)

    # Create replay buffer
    buffer = ReplayBuffer(capacity=1000, obs_dim=3, action_dim=1)
    key, buffer_key = jax.random.split(key)
    buffer_state = buffer.init_buffer_state(buffer_key)

    # Collect some transitions
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key)

    for step in range(50):
        # Select action
        key, action_key = jax.random.split(key)
        action = sac.select_action(sac_state, obs, action_key)

        # Take step
        next_obs, reward, done, next_env_state = env.step(env_state, action)

        # Store transition
        transition = Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        buffer_state = buffer.add(buffer_state, transition)

        # Update for next step
        obs = next_obs
        env_state = next_env_state

    print(f'  Collected {buffer_state.size} transitions')

    # Test training update
    if buffer.can_sample(buffer_state, 32):
        key, sample_key = jax.random.split(key)
        batch = buffer.sample(buffer_state, sample_key, 32)

        key, update_key = jax.random.split(key)
        new_sac_state, info = sac.update_step(sac_state, batch, update_key)

        print(f'  Training update successful! Critic loss: {float(info.get("critic_loss", 0)):.3f}')

    print('✅ Simple Training test passed!')


def main():
    """Run all tests."""
    print('🚀 Running Phase 1 Component Tests')
    print('=' * 50)

    try:
        test_pendulum_environment()
        print()

        test_networks()
        print()

        test_replay_buffer()
        print()

        test_simple_training()
        print()

        print('🎉 All tests passed! Phase 1 implementation is working correctly.')
        print("🔥 You can now run 'python src/main.py' for the full training demo.")

    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
