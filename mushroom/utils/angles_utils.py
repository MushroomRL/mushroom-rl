from math import fmod
import numpy as np


def normalize_angle_positive(angle):
    """
    Wrap the angle between 0 and 2 * pi.

    Args:
        angle (float): angle to wrap.

    Returns:
         The wrapped angle.

    """
    pi_2 = 2.0 * np.pi

    return fmod(fmod(angle, pi_2) + pi_2, pi_2)


def normalize_angle(angle):
    """
    Wrap the angle between -pi and pi.

    Args:
        angle (float): angle to wrap.

    Returns:
         The wrapped angle.

    """
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2.0 * np.pi

    return a


def shortest_angular_distance(from_angle, to_angle):
    return normalize_angle(to_angle - from_angle)
