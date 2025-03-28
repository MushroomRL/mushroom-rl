import numpy as np

from mushroom_rl.utils.angles import mat_to_euler, euler_to_quat


def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm


def quaternion_distance(q1, q2):
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    cos_half_angle = np.abs(np.dot(q1, q2))

    theta = 2 * np.arccos(cos_half_angle)
    return theta / 2


def mat_to_quat(mat):
    euler = mat_to_euler(mat)
    quat = euler_to_quat(euler)
    return quat
