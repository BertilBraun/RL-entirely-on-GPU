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

from environment.cartpole import CartPoleEnv, CartPoleState
from algorithms.replay_buffer import ReplayBuffer, ReplayBufferState, Transition
from algorithms.sac import SAC, AutoAlphaConfig, SACConfig, SACState
from utils.cartpole_viz import CartPoleLiveVisualizer
from utils.training_viz import TrainingVisualizer


# ----------------------------
# Small PyTree dataclasses
# ----------------------------
@chex.dataclass
class TrainCarry:
    """State that persists across chunks (host <-> device boundary)."""

    rng: chex.PRNGKey
    sac_state: SACState
    buffer_state: ReplayBufferState
    env_state: CartPoleState
    obs: chex.Array  # (num_envs, obs_dim)
    env_steps: chex.Array  # (num_envs,) int32
    episode_rewards: chex.Array  # (num_envs,) float32
    total_updates_done: chex.Array  # () int32


@chex.dataclass
class UpdateCarry:
    """Inner carry for per-step parameter updates."""

    rng: chex.PRNGKey
    sac_state: SACState
    buffer_state: ReplayBufferState
    total_updates_done: chex.Array  # () int32
    chunk_updates_done: chex.Array  # () int32
    actor_loss_ema: chex.Array  # () float32
    critic_loss_ema: chex.Array  # () float32
    alpha_ema: chex.Array  # () float32
    q_ema: chex.Array  # () float32


@chex.dataclass
class ChunkCarry:
    """Carry through the scan over env steps inside a chunk."""

    train: TrainCarry
    # meters/EMAs are stored here to avoid tuples
    chunk_updates_done: chex.Array  # () int32
    actor_loss_ema: chex.Array  # () float32
    critic_loss_ema: chex.Array  # () float32
    alpha_ema: chex.Array  # () float32
    q_ema: chex.Array  # () float32
    reward_ema: chex.Array  # () float32


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
    # Algorithm & training config
    # ----------------------------
    sac_cfg = SACConfig(
        learning_rate=3e-4,
        gamma=0.995,
        tau=0.005,
        alpha_config=AutoAlphaConfig(min_alpha=0.03),
        hidden_dims=(128, 128),
    )

    num_envs = 256
    max_episode_steps = 1000
    total_updates = 200_000
    buffer_capacity = 1_000_000
    batch_size = 256
    updates_per_step = num_envs // 4  # network updates per env step
    network_updates_per_gpu_chunk = 1000  # updates per GPU-only chunk
    steps_per_gpu_chunk = (network_updates_per_gpu_chunk + updates_per_step - 1) // updates_per_step
    EMA_BETA = 0.01  # smoothing for meters

    # Host-side options
    enable_training_viz = True
    enable_live_viz = True

    # RNG
    rng = jax.random.PRNGKey(42)

    # ----------------------------
    # Env & agent init (host)
    # ----------------------------
    env = CartPoleEnv(num_envs=num_envs)
    sac = SAC(obs_dim=env.obs_dim, action_dim=env.action_dim, max_action=env.max_force, config=sac_cfg)

    rng, sac_key = jax.random.split(rng)
    sac_state = sac.init_state(sac_key)

    replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=env.obs_dim, action_dim=env.action_dim)
    rng, buf_key = jax.random.split(rng)
    buffer_state = replay_buffer.init_buffer_state(buf_key)

    # Initial obs/state via functional reset
    rng, reset_key = jax.random.split(rng)
    obs0, env_state0 = env.reset(reset_key)

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
        CartPoleLiveVisualizer(num_cartpoles=min(num_envs, 4), length=env.length, rail_limit=env.rail_limit)
        if enable_live_viz
        else None
    )

    print(f'Environment: {num_envs} cart-pole(s)')
    print(f'Network: {sac_cfg.hidden_dims} | LR: {sac_cfg.learning_rate}')
    print(f'Updates: total={total_updates}, per-step={updates_per_step}, per-chunk={network_updates_per_gpu_chunk}')
    print(f'Max episode steps: {max_episode_steps} | Batch size: {batch_size}')
    print('=' * 70)

    # ----------------------------
    # JIT-compiled chunk (GPU-only)
    # ----------------------------
    @jax.jit
    def run_chunk(tc: TrainCarry) -> tuple[TrainCarry, ChunkSummary]:
        """Runs STEPS_PER_CHUNK env steps and up to CHUNK_UPDATES updates on-device."""

        def one_step(c: ChunkCarry, _) -> tuple[ChunkCarry, None]:
            # RNG splits for action & reset (update splits happen in updates loop)
            rng, akey, rkey = jax.random.split(c.train.rng, 3)

            # 1) Action
            action = sac.select_action_stochastic(c.train.sac_state, c.train.obs, akey)

            # 2) Env step
            next_obs, reward, done, next_env_state = env.step(c.train.env_state, action)

            # 3) Auto-reset using provided key
            reset_obs, reset_state = env.reset(rkey)

            env_steps1 = c.train.env_steps + 1
            ep_rew1 = c.train.episode_rewards + reward
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

            # 4) Add transition to buffer
            trans = Transition(obs=c.train.obs, action=action, reward=reward, next_obs=next_obs, done=done)
            buffer_state1 = ReplayBuffer.add_batch(c.train.buffer_state, trans)

            # 5) Parameter updates (always attempt; budgets cap actual count)
            def do_update(ucc: UpdateCarry, _) -> tuple[UpdateCarry, None]:
                next_rng, sk, uk = jax.random.split(ucc.rng, 3)

                batch = ReplayBuffer.sample(ucc.buffer_state, sk, batch_size)
                next_sac_state, info = sac.update_step(ucc.sac_state, batch, uk)

                beta = jnp.asarray(EMA_BETA, jnp.float32)
                return UpdateCarry(
                    rng=next_rng,
                    sac_state=next_sac_state,
                    buffer_state=ucc.buffer_state,
                    total_updates_done=ucc.total_updates_done + 1,
                    chunk_updates_done=ucc.chunk_updates_done + 1,
                    actor_loss_ema=(1 - beta) * ucc.actor_loss_ema + beta * info.actor_info.actor_loss,
                    critic_loss_ema=(1 - beta) * ucc.critic_loss_ema + beta * info.critic_info.q1_loss,
                    alpha_ema=(1 - beta) * ucc.alpha_ema + beta * info.alpha_info.alpha,
                    q_ema=(1 - beta) * ucc.q_ema + beta * info.critic_info.q1_mean,
                ), None

            uc0 = UpdateCarry(
                rng=rng,
                sac_state=c.train.sac_state,
                buffer_state=buffer_state1,
                total_updates_done=c.train.total_updates_done,
                chunk_updates_done=c.chunk_updates_done,
                actor_loss_ema=c.actor_loss_ema,
                critic_loss_ema=c.critic_loss_ema,
                alpha_ema=c.alpha_ema,
                q_ema=c.q_ema,
            )
            uc_f, _ = jax.lax.scan(do_update, uc0, xs=None, length=updates_per_step)

            # Reward EMA across envs
            step_rew_mean = jnp.mean(reward)
            reward_ema2 = 0.99 * c.reward_ema + 0.01 * step_rew_mean

            # Update outer carry
            c2 = ChunkCarry(
                train=TrainCarry(
                    rng=uc_f.rng,
                    sac_state=uc_f.sac_state,
                    buffer_state=uc_f.buffer_state,
                    env_state=env_state1,
                    obs=obs1,
                    env_steps=env_steps1,
                    episode_rewards=ep_rew1,
                    total_updates_done=uc_f.total_updates_done,
                ),
                chunk_updates_done=uc_f.chunk_updates_done,
                actor_loss_ema=uc_f.actor_loss_ema,
                critic_loss_ema=uc_f.critic_loss_ema,
                alpha_ema=uc_f.alpha_ema,
                q_ema=uc_f.q_ema,
                reward_ema=reward_ema2,
            )
            return c2, None

        carry = ChunkCarry(
            train=tc,
            chunk_updates_done=jnp.array(0, jnp.int32),
            actor_loss_ema=jnp.array(0.0, jnp.float32),
            critic_loss_ema=jnp.array(0.0, jnp.float32),
            alpha_ema=jnp.array(0.0, jnp.float32),
            q_ema=jnp.array(0.0, jnp.float32),
            reward_ema=jnp.array(0.0, jnp.float32),
        )

        final_carry, _ = jax.lax.scan(one_step, carry, xs=None, length=steps_per_gpu_chunk)

        summary = ChunkSummary(
            chunk_updates=final_carry.chunk_updates_done,
            actor_loss=final_carry.actor_loss_ema,
            critic_loss=final_carry.critic_loss_ema,
            alpha=final_carry.alpha_ema,
            q=final_carry.q_ema,
            reward=final_carry.reward_ema,
        )
        return final_carry.train, summary

    # ----------------------------
    # Host loop: call run_chunk, then log/viz
    # ----------------------------
    start_time = time.time()

    try:
        while int(train_carry.total_updates_done) < total_updates:
            t0 = time.time()
            train_carry, summary = run_chunk(train_carry)  # GPU-only work

            # Pull tiny scalars
            upd = int(summary.chunk_updates)
            actor_loss = float(summary.actor_loss)
            critic_loss = float(summary.critic_loss)
            alpha = float(summary.alpha)
            qval = float(summary.q)
            rew = float(summary.reward)

            # Host-side viz/log (low frequency)
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

            if live_viz:
                # Lightweight render of first few envs; avoid high-frequency updates
                live_viz.update(
                    train_carry.env_state,
                    episode=int(0),
                    step=int(train_carry.total_updates_done),
                    rewards=jnp.asarray([rew]),
                )

            dt = time.time() - t0
            ups = upd / dt if dt > 0 else 0.0
            print(
                f'+{upd:4d} updates this chunk | {ups:6.1f} upd/s | '
                f'total {int(train_carry.total_updates_done):6d}/{total_updates} | '
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
    if live_viz:
        live_viz.close()


if __name__ == '__main__':
    main()
