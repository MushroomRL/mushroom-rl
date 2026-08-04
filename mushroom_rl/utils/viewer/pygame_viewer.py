import os
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import time
import numpy as np


class ImageViewer:
    """
    Interface to pygame for visualizing plain images.

    """
    def __init__(self, size, dt, headless=False):
        """
        Constructor.

        Args:
            size ([list, tuple]): size of the displayed image;
            dt (float): duration of a control step.
            headless (bool, False): skip the display.

        """
        self._size = size
        self._dt = dt
        self._initialized = False
        self._screen = None
        self._headless = headless

    def display(self, img):
        """
        Display given frame.

        Args:
            img: image to display.

        """
        if self._headless:
            return

        if not self._initialized:
            pygame.init()
            self._initialized = True

        if self._screen is None:
            self._screen = pygame.display.set_mode(self._size)

        img = np.transpose(img, (1, 0, 2))
        surf = pygame.surfarray.make_surface(img)
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        time.sleep(self._dt)

    @property
    def size(self):
        """
        Property.

        Returns:
            The size of the screen.

        """
        return self._size

    def close(self):
        """
        Close the viewer, destroy the window.

        """
        self._screen = None
        pygame.display.quit()


class Viewer:
    """
    Interface to pygame for visualizing mushroom native environments.

    Not only it can visualize images, but also contains many methods to draw primitives in the environment, allowing to
    easily build rendering of environments with a few lines of code.

    """
    def __init__(self, env_width, env_height, min_width=500, min_height=100, max_width=1920,
                 max_height=1080, ideal_scale=None, min_scale=1, background=(0, 0, 0)):
        """
        Constructor.

        The window is sized to fit the environment, keeping the two axes at the same scale so that the drawing is not
        distorted. The scale is raised until the environment fills the minimum window and takes at least
        ``ideal_scale`` pixels per unit, then lowered until it fits the maximum window, each bound acting on its own
        axis. An environment that ends up drawn at fewer than ``min_scale`` pixels per unit is too small to read, so
        a world made of many small units (e.g. the cells of a grid) has ``fits`` set to False rather than being
        shrunk further. An environment smaller than the minimum window is centred on it.

        Args:
            env_width (float): the x dimension limit of the desired environment;
            env_height (float): the y dimension limit of the desired environment;
            min_width (int, 500): the environment is scaled to be at least this wide, in pixels;
            min_height (int, 100): the environment is scaled to be at least this tall, in pixels;
            max_width (int, 1920): the environment is scaled to fit in this width, in pixels;
            max_height (int, 1080): the environment is scaled to fit in this height, in pixels;
            ideal_scale (int, None): the pixels an environment unit takes when the screen has room for them, None to
                ask for no more than the fewest;
            min_scale (int, 1): the fewest pixels an environment unit may take to be drawable;
            background (tuple, (0, 0, 0)): background color of the screen.

        """
        assert min_scale >= 1, 'An environment unit must take at least one pixel.'

        if ideal_scale is None:
            ideal_scale = min_scale

        assert ideal_scale >= min_scale, 'An environment unit cannot ideally take fewer pixels than it may take.'

        scale = max(min_width / env_width, min_height / env_height, ideal_scale)
        scale = min(scale, max_width / env_width, max_height / env_height)

        self._fits = scale >= min_scale

        width = scale * env_width
        height = scale * env_height

        width = int(max(width, min_width))
        height = int(max(height, min_height))

        self._size = (width, height)
        self._width = width
        self._height = height
        self._screen = None
        self._scale = scale
        self._margin = np.array([(width / scale - env_width) / 2, (height / scale - env_height) / 2])
        self._background = background

        self._initialized = False

    @property
    def screen(self):
        """
        Property.

        Returns:
            The screen created by this viewer.

        """
        if not self._initialized:
            pygame.init()
            self._initialized = True

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

    @property
    def fits(self):
        """
        Property.

        Returns:
            Whether the environment fits the screen, and can therefore be drawn.

        """
        return self._fits

    def line(self, start, end, color=(255, 255, 255), width=1):
        """
        Draw a line on the screen.

        Args:
            start (np.ndarray): starting point of the line;
            end (np.ndarray): end point of the line;
            color (tuple (255, 255, 255)): color of the line;
            width (int, 1): width of the line.

        """
        start = self._transform(start)
        end = self._transform(end)

        pygame.draw.line(self.screen, color, start, end, width)

    def grid(self, n_rows, n_columns, color=(255, 255, 255), width=1):
        """
        Draw the lines of a grid of cells over the environment, border included, so that a grid of ``n_rows`` by
        ``n_columns`` unit cells covers the ``[0, n_columns] x [0, n_rows]`` region. The border is drawn just inside
        that region, so that it stays visible when the environment fills the window.

        Args:
            n_rows (int): number of rows of cells;
            n_columns (int): number of columns of cells;
            color (tuple, (255, 255, 255)): color of the lines;
            width (int, 1): width of the lines.

        """
        for row in range(1, n_rows):
            self.line(np.array([0, row]), np.array([n_columns, row]), color, width)

        for column in range(1, n_columns):
            self.line(np.array([column, 0]), np.array([column, n_rows]), color, width)

        top_left = self._transform(np.array([0, n_rows]))
        bottom_right = self._transform(np.array([n_columns, 0]))

        pygame.draw.rect(self.screen, color, pygame.Rect(top_left, bottom_right - top_left), width)

    def square(self, center, angle, edge, color=(255, 255, 255), width=0):
        """
        Draw a square on the screen and apply a roto-translation to it.

        Args:
            center (np.ndarray): the center of the polygon;
            angle (float): the rotation to apply to the polygon;
            edge (float): length of an edge;
            color (tuple, (255, 255, 255)) : the color of the polygon;
            width (int, 0): the width of the polygon line, 0 to fill the
                polygon.

        """
        edge_2 = edge / 2
        points = [[edge_2, edge_2],
                  [edge_2, -edge_2],
                  [-edge_2, -edge_2],
                  [-edge_2, edge_2]]

        self.polygon(center, angle, points, color, width)

    def polygon(self, center, angle, points, color=(255, 255, 255), width=0):
        """
        Draw a polygon on the screen and apply a roto-translation to it.

        Args:
            center (np.ndarray): the center of the polygon;
            angle (float): the rotation to apply to the polygon;
            points (list): the points of the polygon w.r.t. the center;
            color (tuple, (255, 255, 255)) : the color of the polygon;
            width (int, 0): the width of the polygon line, 0 to fill the
                polygon.

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
            center (np.ndarray): the center of the circle;
            radius (float): the radius of the circle;
            color (tuple, (255, 255, 255)): the color of the circle;
            width (int, 0): the width of the circle line, 0 to fill the circle.

        """
        center = self._transform(center)
        radius = int(radius * self._scale)
        pygame.draw.circle(self.screen, color, center, radius, width)

    def arrow_head(self, center, scale, angle, color=(255, 255, 255)):
        """
        Draw an harrow head.

        Args:
            center (np.ndarray): the position of the arrow head;
            scale (float): scale of the arrow, correspond to the length;
            angle (float): the angle of rotation of the angle head;
            color (tuple, (255, 255, 255)): the color of the arrow.

        """
        points = [[-0.5 * scale, -0.5 * scale],
                  [-0.5 * scale, 0.5 * scale],
                  [0.5 * scale, 0]]

        self.polygon(center, angle, points, color)

    def force_arrow(self, center, direction, force, max_force,
                    max_length, color=(255, 255, 255), width=1):
        """
        Draw a force arrow, i.e. an arrow representing a force. The
        length of the arrow is directly proportional to the force value.

        Args:
            center (np.ndarray): the point where the force is applied;
            direction (np.ndarray): the direction of the force;
            force (float): the applied force value;
            max_force (float): the maximum force value;
            max_length (float): the length to use for the maximum force;
            color (tuple, (255, 255, 255)): the color of the arrow;
            width (int, 1): the width of the force arrow.

        """
        length = force / max_force * max_length

        if length != 0:

            c = self._transform(center)
            direction = direction / np.linalg.norm(direction)
            end = center + length * direction
            e = self._transform(end)
            delta = e - c

            pygame.draw.line(self.screen, color, c, e, width)
            self.arrow_head(end, length / 4, -np.arctan2(delta[1], delta[0]),
                            color)
        else:
            self.screen

    def torque_arrow(self, center, torque, max_torque,
                     max_radius, color=(255, 255, 255), width=1):
        """
        Draw a torque arrow, i.e. a circular arrow representing a torque. The
        radius of the arrow is directly proportional to the torque value.

        Args:
            center (np.ndarray): the point where the torque is applied;
            torque (float): the applied torque value;
            max_torque (float): the maximum torque value;
            max_radius (float): the radius to use for the maximum torque;
            color (tuple, (255, 255, 255)): the color of the arrow;
            width (int, 1): the width of the torque arrow.

        """
        angle_end = 3 * np.pi / 2 if torque > 0 else 0
        angle_start = 0 if torque > 0 else np.pi / 2
        radius = abs(torque) / max_torque * max_radius

        r = int(radius * self._scale)
        if r != 0:
            c = self._transform(center)
            rect = pygame.Rect(c[0] - r, c[1] - r, 2 * r, 2 * r)
            pygame.draw.arc(self.screen, color, rect,
                            angle_start, angle_end, width)

            arrow_center = center
            arrow_center[1] -= radius * np.sign(torque)
            arrow_scale = radius / 4
            self.arrow_head(arrow_center, arrow_scale, 0, color)
        else:
            self.screen

    def background_image(self, img):
        """
        Use the given image as background of the environment, rescaling it appropriately. The image covers the
        environment rather than the whole window, so that it stays aligned with everything else that is drawn.

        Args:
            img: the image to be used.

        """
        offset = (self._margin * self._scale).astype(int)
        size = self._size - 2 * offset

        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.smoothscale(surf, size)
        self.screen.blit(surf, offset)

    def function(self, x_s, x_e, f, n_points=100,  width=1, color=(255, 255, 255)):
        """
        Draw the graph of a function in the image.

        Args:
            x_s (float): starting x coordinate;
            x_e (float): final x coordinate;
            f (function): the function that maps x coorinates into y
                coordinates;
            n_points (int, 100): the number of segments used to approximate the
                function to draw;
            width (int, 1): thw width of the line drawn;
            color (tuple, (255,255,255)): the color of the line.

        """
        x = np.linspace(x_s, x_e, n_points)
        y = f(x)

        points = [self._transform([a, b]) for a, b in zip(x, y)]
        pygame.draw.lines(self.screen, color, False, points, width)

    @staticmethod
    def get_frame():
        """
        Getter.

        Returns:
            The current Pygame surface as an RGB array.

        """
        surf = pygame.display.get_surface()
        pygame_frame = pygame.surfarray.array3d(surf)
        frame = pygame_frame.swapaxes(0, 1)

        return frame

    def display(self, s):
        """
        Display current frame and initialize the next frame to the background
        color.

        Args:
            s: time to wait in visualization.

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
        return np.array([(p[0] + self._margin[0]) * self._scale,
                         self._height - (p[1] + self._margin[1]) * self._scale]).astype(int)

    @staticmethod
    def _rotate(p, theta):
        return np.array([np.cos(theta) * p[0] - np.sin(theta) * p[1],
                         np.sin(theta) * p[0] + np.cos(theta) * p[1]])
