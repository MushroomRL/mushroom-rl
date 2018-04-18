import pygame
import time
import numpy as np


class Viewer:
    """
    Interface to pygame for visualizing mushroom native environments.
    """
    def __init__(self, env_width, env_height, width=500, height=500,
                 background=(0, 0, 0)):
        """
        Constructor.
        Args:
            env_width (int): The x dimension limit of the desired environment
            env_height (int): The y dimension limit of the desired environment
            width (int, 500): width of the environment window
            height (int, 500): height of the environment window
            background (tuple, (0, 0, 0)): background color of the screen
        """
        pygame.init()
        self._size = (width, height)
        self._width = width
        self._height = height
        self._screen = None
        self._ratio = np.array([width/env_width, height/env_height])
        self._background = background

    @property
    def screen(self):
        """
        Property.
        Returns:
            The screen created by this viewer.

        """
        if self._screen is None:
            self._screen = pygame.display.set_mode(self._size)

        return self._screen

    @property
    def size(self):
        """
        Property.
        Returns:
            The size of the screen.

        """
        return self._size

    def line(self, start, end, color=(255, 255, 255), width=1):
        """
        Draw a line on the screen.
        Args:
            start (np.ndarray): starting point of the line
            end (np.ndarray): end point of the line
            color (tuple (255, 255, 255)): color of the line
            width (int, 1): width of the line

        """
        start = self._transform(start)
        end = self._transform(end)

        pygame.draw.line(self.screen, color, start, end, width)

    def polygon(self, center, angle, points, color=(255, 255, 255), width=0):
        """
        Draw a polygon on the screen and apply a rototranslation to it.
        Args:
            center (np.ndarray): the center of the polygon
            angle (float): the rotation to apply to the polygon
            points (list): the points of the polygon w.r.t. the center
            color (tuple, (255, 255, 255)) : the color of the polygon
            width (int, 0): the width of the polygon line, 0 to fill the polygon

        """
        poly = list()

        for point in points:
            point = self._rotate(point, angle)
            point += center
            point = self._transform(point)
            poly.append(point)

        pygame.draw.polygon(self.screen, color, poly, width)

    def circle(self, center, radius, color=(255, 255, 255), width=0):
        """
        Draw a circle on the screen.
        Args:
            center (np.ndarray): the center of the circle
            radius (float): the radius of the circle
            color (tuple, (255, 255, 255)) : the color of the circle
            width (int, 0): the width of the circle line, 0 to fill the circle

        """
        center = self._transform(center)
        pygame.draw.circle(self.screen, color, center, radius, width)

    def display(self, s):
        """
        Display current frame and initialize the next frame to the background
        color.
        Args:
            s: time to wait in visualization

        """
        pygame.display.flip()
        time.sleep(s)
        self.screen.fill(self._background)

    def close(self):
        """
        Close the viewer, destroy the window.

        """
        self._screen = None
        pygame.display.quit()

    def _transform(self, p):
        return np.array([p[0]*self._ratio[0],
                         self._height - p[1]*self._ratio[1]])

    def _rotate(self, p, theta):
        return np.array([np.cos(theta)*p[0] - np.sin(theta)*p[1],
                         np.sin(theta)*p[0] + np.cos(theta)*p[1]])
