import os
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import time

from mushroom_rl.utils.icons import icon_path


class PygameWindow:
    """
    Base class for the pygame viewers, managing their window. Once the user closes the window, frames are drawn
    off-screen and are no longer displayed, until :meth:`close` is called.

    """
    _icon_path = icon_path(64)

    def __init__(self, size, software_framebuffer=False):
        """
        Constructor.

        Args:
            size ([list, tuple]): size of the window;
            software_framebuffer (bool, False): open the window without an OpenGL context. Pass True when the
                displayed images are rendered with OpenGL on the same thread.

        """
        self._size = size
        self._software_framebuffer = software_framebuffer
        self._initialized = False
        self._screen = None
        self._closed_by_user = False

    def close(self):
        """
        Close the viewer, destroy the window. The next frame drawn opens a new window.

        """
        self._screen = None
        self._closed_by_user = False
        pygame.display.quit()

    @property
    def screen(self):
        """
        Property. Opens the window, if not open yet.

        Returns:
            The surface to draw on: the window, or an off-screen surface once the user has closed the window.

        """
        if not self._initialized:
            pygame.init()
            self._initialized = True

        if self._screen is None:
            self._screen = self._open_window()

        return self._screen

    @property
    def size(self):
        """
        Property.

        Returns:
            The size of the screen.

        """
        return self._size

    def _show(self, dt):
        if not self._closed_by_user:
            pygame.display.flip()

            if pygame.event.get(pygame.QUIT):
                self.close()
                self._closed_by_user = True
                self._screen = pygame.Surface(self._size)
            else:
                time.sleep(dt)

    def _open_window(self):
        pygame.display.set_icon(pygame.image.load(self._icon_path))

        if self._software_framebuffer:
            previous = os.environ.get('SDL_FRAMEBUFFER_ACCELERATION')
            os.environ['SDL_FRAMEBUFFER_ACCELERATION'] = '0'

            try:
                return pygame.display.set_mode(self._size)
            finally:
                if previous is None:
                    del os.environ['SDL_FRAMEBUFFER_ACCELERATION']
                else:
                    os.environ['SDL_FRAMEBUFFER_ACCELERATION'] = previous
        else:
            return pygame.display.set_mode(self._size)
