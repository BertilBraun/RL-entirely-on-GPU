from typing import Tuple
import matplotlib.pyplot as plt


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
