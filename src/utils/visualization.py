"""
Visualization utilities for cart-pole environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple
import jax.numpy as jnp
import chex


class CartPoleVisualizer:
    """
    Real-time visualization for cart-pole environments.
    """

    def __init__(
        self, num_cartpoles: int = 4, l: float = 1.0, rail_limit: float = 2.0, figsize: Tuple[int, int] = (12, 8)
    ):
        self.num_cartpoles = min(num_cartpoles, 4)  # Limit to 4 for display
        self.l = l  # Pendulum length
        self.rail_limit = rail_limit  # Rail extends from -rail_limit to +rail_limit
        self.figsize = figsize

        # Setup figure and subplots
        if self.num_cartpoles == 1:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
            self.axes = [self.ax]
        elif self.num_cartpoles <= 2:
            self.fig, axes = plt.subplots(1, 2, figsize=figsize)
            self.axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            self.fig, axes = plt.subplots(2, 2, figsize=figsize)
            self.axes = axes.flatten()

        # Initialize cart-pole visualization elements
        self.rails = []
        self.carts = []
        self.pendulum_lines = []
        self.pendulum_bobs = []
        self.trails = []
        self.reward_zones = []

        for i in range(self.num_cartpoles):
            ax = self.axes[i]

            # Set up the coordinate system
            x_margin = 0.5
            y_margin = 0.5
            ax.set_xlim(-rail_limit - x_margin, rail_limit + x_margin)
            ax.set_ylim(-l - y_margin, l + y_margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Cart-Pole {i + 1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Height')

            # Rail (track that cart moves on)
            rail_line = ax.plot([-rail_limit, rail_limit], [0, 0], 'k-', linewidth=4, alpha=0.8)[0]
            self.rails.append(rail_line)

            # Reward zone (top 10% height area)
            reward_height = 0.9 * l
            reward_zone = ax.axhspan(reward_height, l, alpha=0.2, color='green', label='Reward Zone (Top 10%)')
            self.reward_zones.append(reward_zone)

            # Cart (base that moves on rail)
            cart_width = 0.3
            cart_height = 0.15
            cart = plt.Rectangle(
                (-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color='blue', alpha=0.8
            )
            ax.add_patch(cart)
            self.carts.append(cart)

            # Pendulum arm (from cart to bob)
            pendulum_line = ax.plot([], [], 'r-', linewidth=3, alpha=0.8)[0]
            self.pendulum_lines.append(pendulum_line)

            # Pendulum bob
            pendulum_bob = ax.plot([], [], 'ro', markersize=8)[0]
            self.pendulum_bobs.append(pendulum_bob)

            # Trail for bob position
            trail = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)[0]
            self.trails.append(trail)

            # Add legend for first subplot
            if i == 0:
                ax.legend(loc='upper right')

        # Hide unused subplots
        for i in range(self.num_cartpoles, len(self.axes)):
            self.axes[i].set_visible(False)

        plt.tight_layout()

        # Trail data storage
        self.trail_data = [{'x': [], 'y': []} for _ in range(self.num_cartpoles)]
        self.max_trail_length = 100

        # Animation object
        self.animation = None

    def update_cartpoles(self, states) -> None:
        """
        Update cart-pole positions.

        Args:
            states: Array of states [x, x_dot, cos(theta), sin(theta), theta_dot] for each cart-pole
                   or CartPoleState objects with x and theta attributes
        """
        # Handle different input formats
        if hasattr(states, 'x') and hasattr(states, 'theta'):  # CartPoleState object
            x_positions = np.array(states.x)
            theta_values = np.array(states.theta)
        else:
            # Observation array format [x, x_dot, cos(theta), sin(theta), theta_dot]
            states = np.array(states)
            if states.ndim == 1:
                states = states.reshape(1, -1)

            x_positions = states[:, 0]
            cos_theta = states[:, 2]
            sin_theta = states[:, 3]
            theta_values = np.arctan2(sin_theta, cos_theta)

        # Ensure we have the right number of values
        x_positions = x_positions[: self.num_cartpoles]
        theta_values = theta_values[: self.num_cartpoles]

        for i in range(len(x_positions)):
            x = x_positions[i]
            theta = theta_values[i]

            # Update cart position
            self.carts[i].set_x(x - 0.15)  # Center the cart rectangle

            # Calculate pendulum bob position
            bob_x = x + self.l * np.sin(theta)
            bob_y = self.l * np.cos(theta)  # theta=0 is vertical upward

            # Update pendulum arm (from cart center to bob)
            self.pendulum_lines[i].set_data([x, bob_x], [0, bob_y])

            # Update pendulum bob
            self.pendulum_bobs[i].set_data([bob_x], [bob_y])

            # Update trail
            self.trail_data[i]['x'].append(bob_x)
            self.trail_data[i]['y'].append(bob_y)

            # Limit trail length
            if len(self.trail_data[i]['x']) > self.max_trail_length:
                self.trail_data[i]['x'].pop(0)
                self.trail_data[i]['y'].pop(0)

            # Update trail plot
            self.trails[i].set_data(self.trail_data[i]['x'], self.trail_data[i]['y'])

    def clear_trails(self) -> None:
        """Clear all pendulum trails."""
        for i in range(self.num_cartpoles):
            self.trail_data[i] = {'x': [], 'y': []}
            self.trails[i].set_data([], [])

    def show(self, block: bool = True) -> None:
        """Show the visualization."""
        plt.show(block=block)

    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')

    def close(self) -> None:
        """Close the visualization."""
        plt.close(self.fig)


class TrainingVisualizer:
    """
    Visualizer for training metrics and progress.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.metrics_history = {'episode_reward': [], 'actor_loss': [], 'critic_loss': [], 'alpha': [], 'q_values': []}

        # Setup figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.axes = self.axes.flatten()

        self.lines = {}

        # Episode rewards
        ax = self.axes[0]
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        (self.lines['reward'],) = ax.plot([], [], 'b-', alpha=0.7)

        # Losses
        ax = self.axes[1]
        ax.set_title('Training Losses')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        (self.lines['actor_loss'],) = ax.plot([], [], 'r-', alpha=0.7, label='Actor')
        (self.lines['critic_loss'],) = ax.plot([], [], 'g-', alpha=0.7, label='Critic')
        ax.legend()

        # Alpha (temperature)
        ax = self.axes[2]
        ax.set_title('Temperature (Alpha)')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Alpha')
        (self.lines['alpha'],) = ax.plot([], [], 'purple', alpha=0.7)

        # Q-values
        ax = self.axes[3]
        ax.set_title('Q-Values')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Q-Value')
        (self.lines['q_values'],) = ax.plot([], [], 'orange', alpha=0.7)

        plt.tight_layout()

    def update_metrics(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

    def update_plots(self) -> None:
        """Update all plots with current data."""
        # Episode rewards
        if self.metrics_history['episode_reward']:
            episodes = list(range(len(self.metrics_history['episode_reward'])))
            self.lines['reward'].set_data(episodes, self.metrics_history['episode_reward'])
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        # Losses
        if self.metrics_history['actor_loss']:
            steps = list(range(len(self.metrics_history['actor_loss'])))
            self.lines['actor_loss'].set_data(steps, self.metrics_history['actor_loss'])
            self.lines['critic_loss'].set_data(steps, self.metrics_history['critic_loss'])
            self.axes[1].relim()
            self.axes[1].autoscale_view()

        # Alpha
        if self.metrics_history['alpha']:
            steps = list(range(len(self.metrics_history['alpha'])))
            self.lines['alpha'].set_data(steps, self.metrics_history['alpha'])
            self.axes[2].relim()
            self.axes[2].autoscale_view()

        # Q-values
        if self.metrics_history['q_values']:
            steps = list(range(len(self.metrics_history['q_values'])))
            self.lines['q_values'].set_data(steps, self.metrics_history['q_values'])
            self.axes[3].relim()
            self.axes[3].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self, block: bool = True) -> None:
        """Show the visualization."""
        plt.show(block=block)

    def save(self, filename: str) -> None:
        """Save the current plots to file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')

    def close(self) -> None:
        """Close the visualization."""
        plt.close(self.fig)
