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
    def __init__(self, size, dt):
        """
        Constructor.

        Args:
            size ([list, tuple]): size of the displayed image;
            dt (float): duration of a control step.

        """
        self._size = size
        self._dt = dt
        self._initialized = False
        self._screen = None

    def display(self, img):
        """
        Display given frame.

        Args:
            img: image to display.

        """
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

    """
    def __init__(self, env_width, env_height, width=500, height=500,
                 background=(0, 0, 0)):
        """
        Constructor.

        Args:
            env_width (int): The x dimension limit of the desired environment;
            env_height (int): The y dimension limit of the desired environment;
            width (int, 500): width of the environment window;
            height (int, 500): height of the environment window;
            background (tuple, (0, 0, 0)): background color of the screen.

        """
        self._size = (width, height)
        self._width = width
        self._height = height
        self._screen = None
        self._ratio = np.array([width / env_width, height / env_height])
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
        radius = int(radius * self._ratio[0])
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

        r = int(radius * self._ratio[0])
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
        Use the given image as background for the window, rescaling it
        appropriately.

        Args:
            img: the image to be used.

        """
        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.smoothscale(surf, self.size)
        self.screen.blit(surf, (0, 0))

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

        points = [self._transform([a, b]) for a, b in zip(x,y)]
        pygame.draw.lines(self.screen, color, False, points, width)

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
        return np.array([p[0] * self._ratio[0],
                         self._height - p[1] * self._ratio[1]]).astype(int)

    @staticmethod
    def _rotate(p, theta):
        return np.array([np.cos(theta) * p[0] - np.sin(theta) * p[1],
                         np.sin(theta) * p[0] + np.cos(theta) * p[1]])
