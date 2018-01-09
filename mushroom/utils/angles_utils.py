from math import fmod
import numpy as np


def normalize_angle_positive(angle):
    return fmod(fmod(angle, 2.0 * np.pi) + 2.0 * np.pi, 2.0 * np.pi)


def normalize_angle(angle):
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2.0 * np.pi
    return a


def shortest_angular_distance(from_angle, to_angle):
    return normalize_angle(to_angle - from_angle)
