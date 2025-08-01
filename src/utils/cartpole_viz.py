import pygame
import numpy as np
from typing import Tuple

from environment.cartpole import CartPoleState
from utils.base_live_viz import BaseLiveVisualizer, Colors, Point


class CartPoleLiveVisualizer(BaseLiveVisualizer):
    """
    Real-time pygame-based visualization for cart-pole environments during training.
    """

    def __init__(
        self,
        num_cartpoles: int,
        length: float,
        rail_limit: float,
        window_size: Tuple[int, int] = (800, 600),
        should_save: bool = False,
    ):
        super().__init__(num_cartpoles, rail_limit, window_size, should_save)
        self.length = length  # Pendulum length

        # Trail data for pendulum bob
        self.trail_data: list[list[Point]] = [[] for _ in range(self.num_cartpoles)]
        self.max_trail_length = 30

    def update(self, state: CartPoleState, step: int, rewards: np.ndarray) -> None:
        """
        Update the visualization with new cart-pole states.

        Args:
            states: Array of states [x, x_dot, cos(theta), sin(theta), theta_dot] for each cart-pole
            episode_info: Dict with episode, step, reward information

        Returns:
            bool: True if visualization should continue, False if user closed window
        """
        self._base_update(step)

        # Limit to number of cartpoles we're visualizing
        rewards = rewards[: self.num_cartpoles]

        # Draw each cartpole
        for i in range(self.num_cartpoles):
            offset_x, offset_y = self.offsets[i]

            # Extract state information
            x = state.x[i].item()
            cos_theta = np.cos(state.theta[i].item())
            sin_theta = np.sin(state.theta[i].item())
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

            # draw observations below the cartpole
            observations = [
                f'reward: {float(rewards[i]):.2f}',
                f'x: {float(x):.2f}',
                f'x_dot: {float(state.x_dot[i].item()):.2f}',
                f'cos(theta): {float(cos_theta):.2f}',
                f'sin(theta): {float(sin_theta):.2f}',
                f'theta_dot: {float(state.theta_dot[i].item()):.2f}',
            ]
            obs_y = offset_y + self.center_y + 10
            for obs in observations:
                obs_text = self.font.render(obs, True, Colors.text)
                self.screen.blit(obs_text, (offset_x + self.center_x + 10, obs_y))
                obs_y += 25

        self._update_display()

    def clear_trails(self):
        """Clear all pendulum trails."""
        for i in range(self.num_cartpoles):
            self.trail_data[i] = []
