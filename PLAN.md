# 📋 Project Plan: JAX-based SAC for Parallel Pendulum Environments

## ✅ Overall Architecture Goals
- Build a Soft Actor-Critic (SAC) reinforcement learning system for a pendulum environment.
- Use JAX from the beginning for all math, data structures, and neural nets.
- All tensors, buffers, and computation use `jax.numpy` and reside on GPU where possible.
- Final system runs **1000+ parallel pendulum agents**, each mapped to a thread or vector lane on GPU.
- Visualization and observability included from the start.

---

## 🧱 Phase 1: CPU-Compatible JAX Implementation (All JAX APIs)

### 1. Pendulum Physics Simulation (Vectorized)
- Pure JAX function, JIT-compatible.
- Inputs: `(theta, theta_dot, torque)` → Outputs: `(next_theta, next_theta_dot)`
- Batched: `(N,)` for all arrays.
- Integration: forward Euler (GPU-friendly).

```python
def pendulum_step(theta, theta_dot, torque, dt=0.05, g=9.81, l=1.0, m=1.0):
    theta_ddot = (g / l) * jnp.sin(theta) + torque / (m * l**2)
    theta_dot_new = theta_dot + dt * theta_ddot
    theta_new = theta + dt * theta_dot_new
    theta_new = (theta_new + jnp.pi) % (2 * jnp.pi) - jnp.pi  # wrap to [-π, π]
    return theta_new, theta_dot_new
```

---

### 2. State & Reward Functions

* State: `[cos(theta), sin(theta), theta_dot]`
* Reward: `r = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)`

---

### 3. Neural Network Architectures (Flax)

#### Actor Network

* Input: observation `(3,)`
* Output: `(mu, log_std)` for continuous Gaussian action
* Use `tanh` + reparameterization trick

#### Critic Networks (Q1, Q2)

* Input: concatenated `(obs, action)`
* Output: scalar Q-value

---

### 4. SAC Losses (All in JAX)

* Critic loss (TD target using min(Q1, Q2))
* Actor loss (maximize expected Q − α log π)
* Entropy temperature loss (optional)
* Use `optax` for all optimizers

---

### 5. Replay Buffer

* Implement using `jax.numpy` arrays from the beginning
* Circular buffer logic using indexing
* JIT-safe where needed
* Sample mini-batches for training as JAX DeviceArrays

---

### 6. Visualization

* Live rendering of a small number of environments (e.g. 1–4)
* Simple animation using `matplotlib.animation.FuncAnimation` or real-time plot updates
* Used for debugging, loss tracking, and behavior validation

---

## ⚙️ Phase 2: GPU Transition

### 1. JIT All Computation Paths

* Simulation step
* Reward computation
* Policy/value inference
* Loss calculations and gradient updates

### 2. Move Replay Buffer to GPU

* Ensure all data (states, actions, rewards, dones) stay on device
* Sample and update inside JIT-compiled training step

---

### 3. Batched Environment Simulation

* Use `vmap` or manually vectorize across N environments (e.g., N=1024)
* One environment per vector index (conceptually one thread)
* All agents share policy, train jointly from combined buffer
* No Python loops over environments

```python
batched_step = jax.vmap(pendulum_step, in_axes=(0, 0, 0))
```

---

## 🚀 Phase 3: Scalable Multi-Agent Training (Target Architecture)

### 1. Parallel Execution of 1000+ Agents

* All agents execute simulation, observation, reward, and action selection in parallel
* Per-agent state tracked as batched JAX arrays
* Training done with batch samples from all environments

### 2. Fully GPU-Resident Execution

* All computation: sim, networks, replay, sampling, gradient steps remain on GPU
* Minimal host-device sync (e.g., logging, evaluation)

### 3. Batching Strategy

* Replay buffer stores transitions for all agents
* Each agent steps once per environment loop
* Multiple training updates per step if needed
