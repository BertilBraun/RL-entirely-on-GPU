# src/main.py
"""
Main script for JAX-based SAC for Cart-Pole Environment.
Chunked training: do N updates entirely on GPU, then log/viz on host, repeat.
"""

import time
import jax
import jax.numpy as jnp
import chex

from environment.cartpole import CartPoleEnv, CartPoleState
from algorithms.replay_buffer import ReplayBuffer, Transition
from algorithms.sac import SAC, AutoAlphaConfig, SACConfig
from utils.cartpole_viz import CartPoleLiveVisualizer
from utils.training_viz import TrainingVisualizer


# ----------------------------
# Small dataclasses for carry & summary
# ----------------------------
@chex.dataclass
class TrainCarry:
    """State that persists across chunks (host <-> device boundary)."""

    rng: chex.PRNGKey
    sac_state: object
    buffer_state: object
    env_state: CartPoleState
    obs: chex.Array  # (num_envs, obs_dim)
    env_steps: chex.Array  # (num_envs,) int32
    episode_rewards: chex.Array  # (num_envs,) float32
    total_updates_done: chex.Array  # () int32


@chex.dataclass
class ChunkSummary:
    """Small scalar summary returned to the host after each chunk."""

    chunk_updates: chex.Array  # () int32
    actor_loss: chex.Array  # () float32
    critic_loss: chex.Array  # () float32
    alpha: chex.Array  # () float32
    q: chex.Array  # () float32
    reward: chex.Array  # () float32


def main():
    print('🚀 Starting JAX-based SAC for Cart-Pole (Chunked GPU Training)')
    print('=' * 70)

    # ----------------------------
    # Config
    # ----------------------------
    config = SACConfig(
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        hidden_dims=(128, 128),
    )

    # Training params
    num_envs = 256
    max_episode_steps = 1000
    total_updates = 200_000  # Total number of network updates to perform
    buffer_capacity = 100_000
    batch_size = 256
    updates_per_step = num_envs // 4
    CHUNK_UPDATES = 1000  # updates per GPU-only chunk
    STEPS_PER_CHUNK = (CHUNK_UPDATES + updates_per_step - 1) // updates_per_step

    # Host-side viz/log cadence
    enable_training_viz = True
    enable_live_viz = True

    # RNG
    rng = jax.random.PRNGKey(42)

    # ----------------------------
    # Env & agent init (host)
    # ----------------------------
    env = CartPoleEnv(num_envs=num_envs)
    sac = SAC(obs_dim=env.obs_dim, action_dim=env.action_dim, max_action=env.max_force, config=config)

    rng, sac_key = jax.random.split(rng)
    sac_state = sac.init_state(sac_key)

    replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=env.obs_dim, action_dim=env.action_dim)

    rng, buf_key = jax.random.split(rng)
    buffer_state = replay_buffer.init_buffer_state(buf_key)

    # Initial obs/state
    rng, reset_key = jax.random.split(rng)
    obs0, env_state0 = env.reset(reset_key)

    # Carry init
    train_carry = TrainCarry(
        rng=rng,
        sac_state=sac_state,
        buffer_state=buffer_state,
        env_state=env_state0,
        obs=obs0,
        env_steps=jnp.zeros(num_envs, dtype=jnp.int32),
        episode_rewards=jnp.zeros(num_envs, dtype=jnp.float32),
        total_updates_done=jnp.array(0, dtype=jnp.int32),
    )

    # Viz
    training_viz = TrainingVisualizer(figsize=(12, 6)) if enable_training_viz else None
    live_viz = (
        CartPoleLiveVisualizer(num_cartpoles=num_envs, length=env.length, rail_limit=env.rail_limit)
        if enable_live_viz
        else None
    )

    print(f'Environment: {num_envs} cart-pole(s)')
    print(f'Network architecture: {config.hidden_dims}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Total updates: {total_updates} | Chunk updates: {CHUNK_UPDATES}')
    print(f'Max episode steps: {max_episode_steps} | Batch size: {batch_size}')
    print('=' * 70)

    # ----------------------------
    # JIT-compiled chunk (GPU-only)
    # ----------------------------

    @jax.jit
    def run_chunk(tc: TrainCarry) -> tuple[TrainCarry, ChunkSummary]:
        def one_step(carry, _):
            (tc, chunk_updates_done, actor_loss_ema, critic_loss_ema, alpha_ema, q_ema, reward_ema) = carry

            rng = tc.rng
            rng, akey, rkey = jax.random.split(rng, 3)

            # 1) Action
            action = sac.select_action_stochastic(tc.sac_state, tc.obs, akey)

            # 2) Env step
            next_obs, reward, done, next_env_state = env.step(tc.env_state, action)

            # 3) Auto-reset (pure reset)
            reset_obs, reset_state = env.reset(rkey)

            env_steps1 = tc.env_steps + 1
            ep_rew1 = tc.episode_rewards + reward
            should_reset = jnp.logical_or(done, env_steps1 >= max_episode_steps)

            obs1 = jnp.where(should_reset[..., None], reset_obs, next_obs)
            env_state1 = CartPoleState(
                x=jnp.where(should_reset[..., None], reset_state.x, next_env_state.x),
                x_dot=jnp.where(should_reset[..., None], reset_state.x_dot, next_env_state.x_dot),
                theta=jnp.where(should_reset[..., None], reset_state.theta, next_env_state.theta),
                theta_dot=jnp.where(should_reset[..., None], reset_state.theta_dot, next_env_state.theta_dot),
            )
            env_steps1 = jnp.where(should_reset, 0, env_steps1)
            ep_rew1 = jnp.where(should_reset, 0.0, ep_rew1)

            # 4) Add to buffer
            trans = Transition(obs=tc.obs, action=action, reward=reward, next_obs=next_obs, done=done)
            buffer_state1 = ReplayBuffer.add_batch(tc.buffer_state, trans)

            # 5) Do updates conditionally, bounded by global & chunk budget
            def do_updates(args):
                (
                    rng,
                    sac_state,
                    buffer_state,
                    total_updates_done,
                    chunk_updates_done,
                    actor_loss_ema,
                    critic_loss_ema,
                    alpha_ema,
                    q_ema,
                ) = args

                remaining_global = total_updates - total_updates_done
                remaining_chunk = CHUNK_UPDATES - chunk_updates_done
                step_budget = jnp.minimum(updates_per_step, jnp.minimum(remaining_global, remaining_chunk))

                def body(i, inner):
                    (
                        rng,
                        sac_state,
                        buffer_state,
                        total_updates_done,
                        chunk_updates_done,
                        actor_loss_ema,
                        critic_loss_ema,
                        alpha_ema,
                        q_ema,
                    ) = inner

                    def do_one(inner2):
                        (
                            rng,
                            sac_state,
                            buffer_state,
                            total_updates_done,
                            chunk_updates_done,
                            actor_loss_ema,
                            critic_loss_ema,
                            alpha_ema,
                            q_ema,
                        ) = inner2
                        rng, sk = jax.random.split(rng)
                        batch = ReplayBuffer.sample(buffer_state, sk, batch_size)
                        rng, uk = jax.random.split(rng)
                        sac_state, info = sac.update_step(sac_state, batch, uk)

                        beta = 0.01
                        actor_loss_ema = (1 - beta) * actor_loss_ema + beta * info.actor_info.actor_loss
                        critic_loss_ema = (1 - beta) * critic_loss_ema + beta * info.critic_info.q1_loss
                        alpha_ema = (1 - beta) * alpha_ema + beta * info.alpha_info.alpha
                        q_ema = (1 - beta) * q_ema + beta * info.critic_info.q1_mean

                        return (
                            rng,
                            sac_state,
                            buffer_state,
                            total_updates_done + 1,
                            chunk_updates_done + 1,
                            actor_loss_ema,
                            critic_loss_ema,
                            alpha_ema,
                            q_ema,
                        )

                    return jax.lax.cond(
                        i < step_budget,
                        do_one,
                        lambda x: x,
                        (
                            rng,
                            sac_state,
                            buffer_state,
                            total_updates_done,
                            chunk_updates_done,
                            actor_loss_ema,
                            critic_loss_ema,
                            alpha_ema,
                            q_ema,
                        ),
                    )

                return jax.lax.fori_loop(
                    0,
                    updates_per_step,
                    body,
                    (
                        rng,
                        tc.sac_state,
                        buffer_state1,
                        tc.total_updates_done,
                        chunk_updates_done,
                        actor_loss_ema,
                        critic_loss_ema,
                        alpha_ema,
                        q_ema,
                    ),
                )

            can = ReplayBuffer.can_sample(buffer_state1, batch_size)
            (
                rng2,
                sac_state2,
                buffer_state2,
                total_updates_done2,
                chunk_updates_done2,
                actor_loss_ema2,
                critic_loss_ema2,
                alpha_ema2,
                q_ema2,
            ) = jax.lax.cond(
                can,
                do_updates,
                lambda a: a,
                (
                    rng,
                    tc.sac_state,
                    buffer_state1,
                    tc.total_updates_done,
                    chunk_updates_done,
                    actor_loss_ema,
                    critic_loss_ema,
                    alpha_ema,
                    q_ema,
                ),
            )

            # Reward EMA across envs
            step_rew_mean = jnp.mean(reward)
            reward_ema2 = 0.99 * reward_ema + 0.01 * step_rew_mean

            # Update carry
            tc2 = TrainCarry(
                rng=rng2,
                sac_state=sac_state2,
                buffer_state=buffer_state2,
                env_state=env_state1,
                obs=obs1,
                env_steps=env_steps1,
                episode_rewards=ep_rew1,
                total_updates_done=total_updates_done2,
            )

            new_carry = (
                tc2,
                chunk_updates_done2,
                actor_loss_ema2,
                critic_loss_ema2,
                alpha_ema2,
                q_ema2,
                reward_ema2,
            )
            return new_carry, None

        # Per-chunk accumulators
        chunk_updates_done = jnp.array(0, jnp.int32)
        actor_loss_ema = jnp.array(0.0, jnp.float32)
        critic_loss_ema = jnp.array(0.0, jnp.float32)
        alpha_ema = jnp.array(0.0, jnp.float32)
        q_ema = jnp.array(0.0, jnp.float32)
        reward_ema = jnp.array(0.0, jnp.float32)

        init = (tc, chunk_updates_done, actor_loss_ema, critic_loss_ema, alpha_ema, q_ema, reward_ema)
        final, _ = jax.lax.scan(one_step, init, xs=None, length=STEPS_PER_CHUNK)

        (
            tc_f,
            chunk_updates_done_f,
            actor_loss_ema_f,
            critic_loss_ema_f,
            alpha_ema_f,
            q_ema_f,
            reward_ema_f,
        ) = final

        summary = ChunkSummary(
            chunk_updates=chunk_updates_done_f,
            actor_loss=actor_loss_ema_f,
            critic_loss=critic_loss_ema_f,
            alpha=alpha_ema_f,
            q=q_ema_f,
            reward=reward_ema_f,
        )
        return tc_f, summary

    # ----------------------------
    # Host loop: call run_chunk, then log/viz
    # ----------------------------
    start_time = time.time()
    wall_updates = 0

    try:
        while int(train_carry.total_updates_done) < total_updates:
            t0 = time.time()
            train_carry, summary = run_chunk(train_carry)  # GPU-only work

            # Pull small scalars
            upd = int(summary.chunk_updates)
            wall_updates += upd
            actor_loss = float(summary.actor_loss)
            critic_loss = float(summary.critic_loss)
            alpha = float(summary.alpha)
            qval = float(summary.q)
            rew = float(summary.reward)

            # Viz/log
            if training_viz:
                training_viz.update_metrics(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    alpha=alpha,
                    q_values=qval,
                    episode_reward=rew,
                )
                training_viz.update_plots()
                training_viz.show(block=False)

            dt = time.time() - t0
            ups = upd / dt if dt > 0 else 0.0
            total_ups = int(train_carry.total_updates_done)
            print(
                f'+{upd:4d} updates this chunk | {ups:6.1f} upd/s | '
                f'total {total_ups:6d}/{total_updates} | '
                f'actor {actor_loss:.3f} | critic {critic_loss:.3f} | '
                f'alpha {alpha:.3f} | q {qval:.2f} | r {rew:.2f}'
            )

    except KeyboardInterrupt:
        print('\n⏸️  Training interrupted by user')

    # ----------------------------
    # Final stats & cleanup
    # ----------------------------
    elapsed_time = time.time() - start_time
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


if __name__ == '__main__':
    main()
