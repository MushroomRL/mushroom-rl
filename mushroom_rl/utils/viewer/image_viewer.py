import pygame
import numpy as np

from mushroom_rl.utils.viewer.pygame_window import PygameWindow


class ImageViewer(PygameWindow):
    """
    Interface to pygame for visualizing plain images.

    """
    def __init__(self, size, dt, headless=False, software_framebuffer=False):
        """
        Constructor.

        Args:
            size ([list, tuple]): size of the displayed image;
            dt (float): duration of a control step.
            headless (bool, False): skip the display;
            software_framebuffer (bool, False): open the window without an OpenGL context. Pass True when the
                displayed images are rendered with OpenGL on the same thread.

        """
        super().__init__(size, software_framebuffer)

        self._dt = dt
        self._headless = headless

    def display(self, img):
        """
        Display given frame.

        Args:
            img: image to display.

        """
        if not self._headless:
            img = np.transpose(img, (1, 0, 2))
            surf = pygame.surfarray.make_surface(img)
            self.screen.blit(surf, (0, 0))
            self._show(self._dt)
