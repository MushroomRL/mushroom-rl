import numpy as np
import pytest
from mushroom_rl.utils.angles import (
    normalize_angle_positive, normalize_angle, shortest_angular_distance,
    quat_to_euler, euler_to_quat, mat_to_euler, euler_to_mat
)
from mushroom_rl.utils.quaternions import (
    normalize_quaternion, quaternion_distance, mat_to_quat
)


def test_normalize_angle_positive():
    assert np.isclose(normalize_angle_positive(0.), 0.)
    assert np.isclose(normalize_angle_positive(2 * np.pi), 0., atol=1e-10)
    assert np.isclose(normalize_angle_positive(-np.pi), np.pi)
    assert np.isclose(normalize_angle_positive(3 * np.pi), np.pi)


def test_normalize_angle():
    assert np.isclose(normalize_angle(0.), 0.)
    assert np.isclose(normalize_angle(np.pi), np.pi)
    assert np.isclose(normalize_angle(-np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(normalize_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(normalize_angle(2 * np.pi + 0.5), 0.5)


def test_shortest_angular_distance():
    assert np.isclose(shortest_angular_distance(0., np.pi / 2), np.pi / 2)
    assert np.isclose(shortest_angular_distance(np.pi / 2, 0.), -np.pi / 2)
    assert np.isclose(shortest_angular_distance(0., 3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(shortest_angular_distance(0., 0.), 0.)


def test_euler_quat_roundtrip():
    np.random.seed(42)
    for _ in range(10):
        euler = np.random.uniform(-np.pi / 2, np.pi / 2, 3)
        quat = euler_to_quat(euler)
        euler_back = quat_to_euler(quat)
        assert np.allclose(euler, euler_back, atol=1e-10)


def test_euler_quat_batch():
    np.random.seed(42)
    euler = np.random.uniform(-np.pi / 4, np.pi / 4, (3, 5))
    quat = euler_to_quat(euler)
    assert quat.shape == (4, 5)
    euler_back = quat_to_euler(quat)
    assert np.allclose(euler, euler_back, atol=1e-10)


def test_mat_to_euler():
    euler = np.array([0.1, 0.2, 0.3])
    mat = euler_to_mat(euler)
    euler_back = mat_to_euler(mat)
    assert np.allclose(euler, euler_back, atol=1e-10)


def test_euler_to_mat():
    from scipy.spatial.transform import Rotation as R
    euler = np.array([0.3, -0.1, 0.5])
    mat = euler_to_mat(euler)
    expected = R.from_euler('xyz', euler).as_matrix()
    assert np.allclose(mat, expected)


def test_normalize_quaternion():
    q = np.array([2., 0., 0., 0.])
    q_norm = normalize_quaternion(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)
    assert np.allclose(q_norm, [1., 0., 0., 0.])


def test_quaternion_distance_same():
    q = np.array([1., 0., 0., 0.])
    assert np.isclose(quaternion_distance(q, q), 0.)


def test_quaternion_distance_opposite():
    q1 = np.array([1., 0., 0., 0.])
    q2 = np.array([-1., 0., 0., 0.])
    # opposite quaternions represent the same rotation → distance = 0
    assert np.isclose(quaternion_distance(q1, q2), 0.)


def test_quaternion_distance_90_deg():
    # 90-degree rotation around z: q = [cos(45°), 0, 0, sin(45°)]
    q1 = np.array([1., 0., 0., 0.])
    q2 = np.array([np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)])
    dist = quaternion_distance(q1, q2)
    assert np.isclose(dist, np.pi / 4)


def test_mat_to_quat():
    euler = np.array([0.1, 0.2, 0.3])
    mat = euler_to_mat(euler)
    quat = mat_to_quat(mat)
    euler_back = quat_to_euler(quat)
    assert np.allclose(euler, euler_back, atol=1e-10)