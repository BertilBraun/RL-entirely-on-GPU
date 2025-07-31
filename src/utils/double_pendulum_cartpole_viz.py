import pygame
import numpy as np
from PIL import Image
from typing import Tuple
from dataclasses import dataclass

from environment.double_pendulum_cartpole import DoublePendulumCartPoleState


@dataclass
class Point:
    x: float
    y: float


class Colors:
    background = (240, 240, 240)
    rail = (50, 50, 200)
    cart = (50, 100, 200)
    pendulum1 = (200, 50, 50)  # First pendulum (red)
    pendulum2 = (50, 200, 50)  # Second pendulum (green)
    bob1 = (150, 50, 50)  # First bob (dark red)
    bob2 = (50, 150, 50)  # Second bob (dark green)
    trail1 = (200, 50, 50, 100)  # First pendulum trail
    trail2 = (50, 200, 50, 100)  # Second pendulum trail
    text = (0, 0, 0)
    reward_zone = (50, 200, 50, 80)


class DoublePendulumCartPoleLiveVisualizer:
    """
    Real-time pygame-based visualization for double pendulum cart-pole environments during training.
    """

    def __init__(
        self,
        num_cartpoles: int,
        length1: float,
        length2: float,
        rail_limit: float,
        window_size: Tuple[int, int] = (800, 600),
        should_save: bool = False,
    ) -> None:
        self.num_cartpoles = min(num_cartpoles, 4)  # Limit to 4 for display
        self.length1 = length1  # First pendulum length
        self.length2 = length2  # Second pendulum length
        self.rail_limit = rail_limit
        self.window_size = window_size
        self.should_save = should_save
        self.frames = []

        # Initialize pygame
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Double Pendulum Cart-Pole Live Training')

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

        # Calculate display scaling (use max length for scaling)
        max_length = max(length1, length2)
        total_length = length1 + length2  # Maximum reach of double pendulum
        elements_on_x = 2 if self.num_cartpoles > 1 else 1
        elements_on_y = 2 if self.num_cartpoles > 2 else 1
        scale_x = window_size[0] / (2 * rail_limit * elements_on_x)
        scale_y = window_size[1] / (2 * total_length * elements_on_y)
        self.scale = min(scale_x, scale_y) * 0.8
        self.center_x = window_size[0] // 2
        self.center_y = window_size[1] // 2

        # Trail data for both pendulum bobs
        self.trail1_data: list[list[Point]] = [[] for _ in range(self.num_cartpoles)]
        self.trail2_data: list[list[Point]] = [[] for _ in range(self.num_cartpoles)]
        self.max_trail_length = 50

        # Font for text
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.center_x + x * self.scale)
        screen_y = int(self.center_y - y * self.scale)  # Flip Y axis
        return screen_x, screen_y

    def update(self, state: DoublePendulumCartPoleState, step: int, rewards: np.ndarray) -> None:
        """
        Update the visualization with new double pendulum cart-pole states.

        Args:
            state: DoublePendulumCartPoleState with x, x_dot, theta1, theta1_dot, theta2, theta2_dot
            step: Current step number
            rewards: Array of rewards for each environment
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Clear screen
        self.screen.fill(Colors.background)

        # Limit to number of cartpoles we're visualizing
        rewards = rewards[: self.num_cartpoles]

        # Draw each double pendulum cartpole
        for i in range(self.num_cartpoles):
            offset_x, offset_y = self.offsets[i]

            # Extract state information
            x = float(state.x[i].item())
            theta1 = float(state.theta1[i].item())
            theta2 = float(state.theta2[i].item())

            # Calculate positions with offset
            cart_screen_x, cart_screen_y = self.world_to_screen(x, 0)
            cart_screen_x += offset_x
            cart_screen_y += offset_y

            # Calculate first pendulum bob position
            bob1_x = x + self.length1 * np.sin(theta1)
            bob1_y = self.length1 * np.cos(theta1)  # theta=0 is vertical upward
            bob1_screen_x, bob1_screen_y = self.world_to_screen(bob1_x, bob1_y)
            bob1_screen_x += offset_x
            bob1_screen_y += offset_y

            # Calculate second pendulum bob position (relative to first bob)
            bob2_x = bob1_x + self.length2 * np.sin(theta2)
            bob2_y = bob1_y + self.length2 * np.cos(theta2)
            bob2_screen_x, bob2_screen_y = self.world_to_screen(bob2_x, bob2_y)
            bob2_screen_x += offset_x
            bob2_screen_y += offset_y

            # Draw reward zone (top area where both pendulums should be)
            total_length = self.length1 + self.length2
            reward_height = 0.9 * total_length
            reward_zone_start = self.world_to_screen(-self.rail_limit, reward_height)
            reward_zone_end = self.world_to_screen(self.rail_limit, total_length)
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

            # Update trails for both bobs
            self.trail1_data[i].append(Point(x=bob1_screen_x, y=bob1_screen_y))
            if len(self.trail1_data[i]) > self.max_trail_length:
                self.trail1_data[i].pop(0)

            self.trail2_data[i].append(Point(x=bob2_screen_x, y=bob2_screen_y))
            if len(self.trail2_data[i]) > self.max_trail_length:
                self.trail2_data[i].pop(0)

            # Draw trails
            if len(self.trail1_data[i]) > 1:
                pygame.draw.lines(
                    self.screen,
                    Colors.trail1[:3],
                    False,
                    [(point.x, point.y) for point in self.trail1_data[i]],
                    2,
                )

            if len(self.trail2_data[i]) > 1:
                pygame.draw.lines(
                    self.screen,
                    Colors.trail2[:3],
                    False,
                    [(point.x, point.y) for point in self.trail2_data[i]],
                    2,
                )

            # Draw first pendulum arm (cart to first bob)
            pygame.draw.line(
                self.screen, Colors.pendulum1, (cart_screen_x, cart_screen_y), (bob1_screen_x, bob1_screen_y), 4
            )

            # Draw second pendulum arm (first bob to second bob)
            pygame.draw.line(
                self.screen, Colors.pendulum2, (bob1_screen_x, bob1_screen_y), (bob2_screen_x, bob2_screen_y), 4
            )

            # Draw pendulum bobs
            pygame.draw.circle(self.screen, Colors.bob1, (bob1_screen_x, bob1_screen_y), 10)
            pygame.draw.circle(self.screen, Colors.bob2, (bob2_screen_x, bob2_screen_y), 8)

            # Draw observations below the cartpole
            observations = [
                f'reward: {float(rewards[i]):.2f}',
                f'x: {float(x):.2f}',
                f'x_dot: {float(state.x_dot[i].item()):.2f}',
                f'theta1: {float(theta1):.2f}',
                f'theta1_dot: {float(state.theta1_dot[i].item()):.2f}',
                f'theta2: {float(theta2):.2f}',
                f'theta2_dot: {float(state.theta2_dot[i].item()):.2f}',
            ]
            obs_y = offset_y + self.center_y + 10
            for obs in observations:
                obs_text = self.font.render(obs, True, Colors.text)
                self.screen.blit(obs_text, (offset_x + self.center_x + 10, obs_y))
                obs_y += 22

        # Draw episode information
        info_y = 10
        info_texts = [f'Step: {step}']

        for text_str in info_texts:
            text = self.font.render(text_str, True, Colors.text)
            self.screen.blit(text, (10, info_y))
            info_y += 25

        # Update display
        pygame.display.flip()
        if self.should_save:
            x3 = pygame.surfarray.array3d(self.screen)
            x3 = np.moveaxis(x3, 0, 1)
            array = Image.fromarray(np.uint8(x3))
            self.frames.append(array)

    def clear_trails(self) -> None:
        """Clear all pendulum trails."""
        for i in range(self.num_cartpoles):
            self.trail1_data[i] = []
            self.trail2_data[i] = []

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()

    def save_frames(self, path: str, fps: int = 60) -> None:
        """Save the frames to a gif file."""
        if not self.should_save:
            raise ValueError('Frames not saved because should_save is False')
        if not self.frames:
            raise ValueError('No frames to save')

        self.frames[0].save(
            path,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / fps),  # display time of each frame in ms
        )
