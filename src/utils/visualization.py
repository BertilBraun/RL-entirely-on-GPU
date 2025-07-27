"""
Visualization utilities for pendulum environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple
import jax.numpy as jnp
import chex


class PendulumVisualizer:
    """
    Real-time visualization for pendulum environments.
    """

    def __init__(self, num_pendulums: int = 4, l: float = 1.0, figsize: Tuple[int, int] = (10, 8)):
        self.num_pendulums = min(num_pendulums, 4)  # Limit to 4 for display
        self.l = l  # Pendulum length
        self.figsize = figsize

        # Setup figure and subplots
        if self.num_pendulums == 1:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
            self.axes = [self.ax]
        elif self.num_pendulums <= 2:
            self.fig, axes = plt.subplots(1, 2, figsize=figsize)
            self.axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            self.fig, axes = plt.subplots(2, 2, figsize=figsize)
            self.axes = axes.flatten()

        # Initialize pendulum lines and points
        self.lines = []
        self.points = []
        self.trails = []

        for i in range(self.num_pendulums):
            ax = self.axes[i]
            ax.set_xlim(-1.5 * l, 1.5 * l)
            ax.set_ylim(-1.5 * l, 1.5 * l)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Pendulum {i + 1}')

            # Pendulum arm
            (line,) = ax.plot([], [], 'b-', linewidth=3, alpha=0.8)
            self.lines.append(line)

            # Pendulum bob
            (point,) = ax.plot([], [], 'ro', markersize=10)
            self.points.append(point)

            # Trail for bob position
            (trail,) = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)
            self.trails.append(trail)

        # Hide unused subplots
        for i in range(self.num_pendulums, len(self.axes)):
            self.axes[i].set_visible(False)

        plt.tight_layout()

        # Trail data storage
        self.trail_data = [{'x': [], 'y': []} for _ in range(self.num_pendulums)]
        self.max_trail_length = 100

        # Animation object
        self.animation = None

    def update_pendulums(self, theta_values: chex.Array) -> None:
        """
        Update pendulum positions.

        Args:
            theta_values: Array of theta values for each pendulum
        """
        # Convert to numpy if needed
        if hasattr(theta_values, 'shape'):
            thetas = np.array(theta_values)
        else:
            thetas = theta_values

        # Ensure we have the right number of values
        if thetas.ndim == 0:
            thetas = np.array([thetas])
        thetas = thetas[: self.num_pendulums]

        for i, theta in enumerate(thetas):
            # Calculate pendulum position
            x = self.l * np.sin(theta)
            y = -self.l * np.cos(theta)  # Negative because y-axis is flipped

            # Update pendulum arm (from origin to bob)
            self.lines[i].set_data([0, x], [0, y])

            # Update pendulum bob
            self.points[i].set_data([x], [y])

            # Update trail
            self.trail_data[i]['x'].append(x)
            self.trail_data[i]['y'].append(y)

            # Limit trail length
            if len(self.trail_data[i]['x']) > self.max_trail_length:
                self.trail_data[i]['x'].pop(0)
                self.trail_data[i]['y'].pop(0)

            # Update trail plot
            self.trails[i].set_data(self.trail_data[i]['x'], self.trail_data[i]['y'])

    def clear_trails(self) -> None:
        """Clear all pendulum trails."""
        for i in range(self.num_pendulums):
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
