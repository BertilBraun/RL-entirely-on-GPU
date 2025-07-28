from dataclasses import dataclass
import pygame
import numpy as np
from typing import Tuple


@dataclass
class EpisodeInfo:
    episode: int
    step: int
    reward: float


@dataclass
class Point:
    x: float
    y: float


class Colors:
    background = (240, 240, 240)
    rail = (50, 50, 200)
    cart = (50, 100, 200)
    pendulum = (200, 50, 50)
    bob = (150, 50, 50)
    trail = (200, 50, 50, 100)
    text = (0, 0, 0)
    reward_zone = (50, 200, 50, 80)


class CartPoleLiveVisualizer:
    """
    Real-time pygame-based visualization for cart-pole environments during training.
    """

    def __init__(self, num_cartpoles: int, length: float, rail_limit: float, window_size: Tuple[int, int] = (800, 600)):
        self.num_cartpoles = min(num_cartpoles, 4)  # Limit to 4 for display
        self.length = length  # Pendulum length
        self.rail_limit = rail_limit
        self.window_size = window_size

        # Initialize pygame
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Cart-Pole Live Training')

        # Calculate layout for multiple cartpoles
        if self.num_cartpoles == 1:
            self.offsets = [(0, 0)]
        elif self.num_cartpoles == 2:
            self.offsets = [(-self.window_size[0] // 4, 0), (self.window_size[0] // 4, 0)]
        elif self.num_cartpoles <= 4:
            self.offsets = [
                (-self.window_size[0] // 4, -self.window_size[1] // 4),
                (self.window_size[0] // 4, -self.window_size[1] // 4),
                (-self.window_size[0] // 4, self.window_size[1] // 4),
                (self.window_size[0] // 4, self.window_size[1] // 4),
            ]
        else:
            raise ValueError(f'Number of cartpoles must be between 1 and 4, got {self.num_cartpoles}')

        # Calculate display scaling
        elements_on_x = 2 if self.num_cartpoles > 1 else 1
        elements_on_y = 2 if self.num_cartpoles > 2 else 1
        scale_x = window_size[0] / (2 * rail_limit * elements_on_x)
        scale_y = window_size[1] / (2 * length * elements_on_y)
        self.scale = min(scale_x, scale_y) * 0.8
        self.center_x = window_size[0] // 2
        self.center_y = window_size[1] // 2

        # Trail data for pendulum bob
        self.trail_data: list[list[Point]] = [[] for _ in range(self.num_cartpoles)]
        self.max_trail_length = 50

        # Font for text
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 24)

        # Episode info
        self.episode_info = EpisodeInfo(episode=0, step=0, reward=0.0)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.center_x + x * self.scale)
        screen_y = int(self.center_y - y * self.scale)  # Flip Y axis
        return screen_x, screen_y

    def update(self, states: np.ndarray, episode: int, step: int, reward: float) -> None:
        """
        Update the visualization with new cart-pole states.

        Args:
            states: Array of states [x, x_dot, cos(theta), sin(theta), theta_dot] for each cart-pole
            episode_info: Dict with episode, step, reward information

        Returns:
            bool: True if visualization should continue, False if user closed window
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Update episode info
        self.episode_info.episode = episode
        self.episode_info.step = step
        self.episode_info.reward = reward

        # Clear screen
        self.screen.fill(Colors.background)

        # Ensure states is 2D
        states = np.array(states)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        # Limit to number of cartpoles we're visualizing
        states = states[: self.num_cartpoles]

        # Draw each cartpole
        for i, state in enumerate(states):
            offset_x, offset_y = self.offsets[i]

            # Extract state information
            x = state[0]
            cos_theta = state[2]
            sin_theta = state[3]
            theta = np.arctan2(sin_theta, cos_theta)

            # Calculate positions with offset
            cart_screen_x, cart_screen_y = self.world_to_screen(x, 0)
            cart_screen_x += offset_x
            cart_screen_y += offset_y

            # Calculate pendulum bob position
            bob_x = x + self.length * np.sin(theta)
            bob_y = self.length * np.cos(theta)  # theta=0 is vertical upward
            bob_screen_x, bob_screen_y = self.world_to_screen(bob_x, bob_y)
            bob_screen_x += offset_x
            bob_screen_y += offset_y

            # Draw reward zone (top 10% height area)
            reward_height = 0.9 * self.length
            reward_zone_start = self.world_to_screen(-self.rail_limit, reward_height)
            reward_zone_end = self.world_to_screen(self.rail_limit, self.length)
            reward_zone_width = reward_zone_end[0] - reward_zone_start[0]
            reward_zone_height = reward_zone_end[1] - reward_zone_start[1]

            # Create reward zone surface with transparency
            reward_surface = pygame.Surface((reward_zone_width, abs(reward_zone_height)))
            reward_surface.set_alpha(80)
            reward_surface.fill(Colors.reward_zone[:3])
            self.screen.blit(
                reward_surface,
                (reward_zone_start[0] + offset_x, min(reward_zone_start[1], reward_zone_end[1]) + offset_y),
            )

            # Draw rail
            rail_start = self.world_to_screen(-self.rail_limit, 0)
            rail_end = self.world_to_screen(self.rail_limit, 0)
            pygame.draw.line(
                self.screen,
                Colors.rail,
                (rail_start[0] + offset_x, rail_start[1] + offset_y),
                (rail_end[0] + offset_x, rail_end[1] + offset_y),
                4,
            )

            # Draw cart
            cart_width = int(0.3 * self.scale)
            cart_height = int(0.15 * self.scale)
            cart_rect = pygame.Rect(
                cart_screen_x - cart_width // 2, cart_screen_y - cart_height // 2, cart_width, cart_height
            )
            pygame.draw.rect(self.screen, Colors.cart, cart_rect)

            # Update trail
            self.trail_data[i].append(Point(x=bob_screen_x, y=bob_screen_y))
            if len(self.trail_data[i]) > self.max_trail_length:
                self.trail_data[i].pop(0)

            # Draw trail
            if len(self.trail_data[i]) > 1:
                pygame.draw.lines(
                    self.screen,
                    Colors.trail[:3],
                    False,
                    [(point.x, point.y) for point in self.trail_data[i]],
                    2,
                )

            # Draw pendulum arm
            pygame.draw.line(
                self.screen, Colors.pendulum, (cart_screen_x, cart_screen_y), (bob_screen_x, bob_screen_y), 3
            )

            # Draw pendulum bob
            pygame.draw.circle(self.screen, Colors.bob, (bob_screen_x, bob_screen_y), 8)

        # Draw episode information
        info_y = 10
        info_texts = [
            f'Episode: {self.episode_info.episode}',
            f'Step: {self.episode_info.step}',
            f'Reward: {self.episode_info.reward:.2f}',
        ]

        for text_str in info_texts:
            text = self.font.render(text_str, True, Colors.text)
            self.screen.blit(text, (10, info_y))
            info_y += 25

        # Update display
        pygame.display.flip()

    def clear_trails(self):
        """Clear all pendulum trails."""
        for i in range(self.num_cartpoles):
            self.trail_data[i] = []

    def close(self):
        """Close the pygame window."""
        pygame.quit()
