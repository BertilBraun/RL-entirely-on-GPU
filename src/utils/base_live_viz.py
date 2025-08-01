import pygame
import numpy as np
from PIL import Image
from typing import Any, Tuple
from dataclasses import dataclass


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


class BaseLiveVisualizer:
    """
    Real-time pygame-based visualization for environments during training.
    """

    def __init__(
        self,
        num_cartpoles: int,
        rail_limit: float,
        window_size: Tuple[int, int] = (800, 600),
        should_save: bool = False,
    ) -> None:
        self.num_cartpoles = min(num_cartpoles, 4)  # Limit to 4 for display
        self.rail_limit = rail_limit
        self.window_size = window_size
        self.should_save = should_save
        self.frames = []

        # Initialize pygame
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Live Training')

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
        elements_on_x = 2 if self.num_cartpoles > 1 else 1
        elements_on_y = 2 if self.num_cartpoles > 2 else 1
        scale_x = window_size[0] / (2 * rail_limit * elements_on_x)
        scale_y = window_size[1] / (2 * rail_limit * elements_on_y)
        self.scale = min(scale_x, scale_y) * 0.8
        self.center_x = window_size[0] // 2
        self.center_y = window_size[1] // 2

        # Font for text
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.center_x + x * self.scale)
        screen_y = int(self.center_y - y * self.scale)  # Flip Y axis
        return screen_x, screen_y

    def _base_update(self, step: int) -> None:
        """
        Update the visualization with new states.
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Clear screen
        self.screen.fill(Colors.background)

        # Draw episode information
        info_y = 10
        info_texts = [f'Step: {step}']

        for text_str in info_texts:
            text = self.font.render(text_str, True, Colors.text)
            self.screen.blit(text, (10, info_y))
            info_y += 25

    def _update_display(self) -> None:
        # Update display
        pygame.display.flip()
        if self.should_save:
            x3 = pygame.surfarray.array3d(self.screen)
            x3 = np.moveaxis(x3, 0, 1)
            array = Image.fromarray(np.uint8(x3))
            self.frames.append(array)

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()

    def save_frames(self, path: str, fps: int) -> None:
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
