from scipy.spatial.transform import Rotation as R
import numpy as np


def quat_to_euler(quat, **kwargs):
    # quat must be in format [w, x, y, z]
    # Conversion from rad to rads
    if len(quat.shape) < 2:
        return R.from_quat(quat[[1, 2, 3, 0]], **kwargs).as_euler('xyz', **kwargs)
    else:
        return R.from_quat(quat[[1, 2, 3, 0], :].T, **kwargs).as_euler('xyz', **kwargs).T


def euler_to_quat(euler, **kwargs):
    if len(euler.shape) < 2:
        return R.from_euler('xyz', euler, **kwargs).as_quat()[[3, 0, 1, 2]]
    else:
        return R.from_euler('xyz', euler.T, **kwargs).as_quat()[:, [3, 0, 1, 2]].T


def convert_traj_euler_to_quat(euler_traj, offset=0):
    # humanoid trajectory conversion -> euler to quaternion
    # offset can be used if passing traj in which the first
    # X observations have been removed
    euler_traj = euler_traj.copy()
    is_vect = len(euler_traj.shape) < 2

    if is_vect:
        quat_traj = np.zeros((euler_traj.shape[0] + 1, ))
        quat_traj[0       :3-offset] = euler_traj[0:3-offset]
        quat_traj[3-offset:7-offset] = euler_to_quat(euler_traj[3-offset:6-offset])
        quat_traj[7-offset:        ] = euler_traj[6-offset:]
    else:
        quat_traj = np.zeros((euler_traj.shape[0] + 1, euler_traj.shape[1]))
        quat_traj[0       :3-offset, :] = euler_traj[0:3-offset, :]
        quat_traj[3-offset:7-offset, :] = euler_to_quat(euler_traj[3-offset:6-offset, :])
        quat_traj[7-offset:,         :] = euler_traj[6-offset:, :]

    return quat_traj


def convert_traj_quat_to_euler(quat_traj, offset=0):
    # humanoid trajectory conversion -> quaternion to euler
    # offset can be used if passing traj in which the first
    # X observations have been removed
    quat_traj = quat_traj.copy()
    is_vect = len(quat_traj.shape) < 2

    if is_vect:
        euler_traj = np.zeros((quat_traj.shape[0] - 1, ))
        euler_traj[0         :3-offset] = quat_traj[0:3-offset]
        euler_traj[3-offset:6-offset] = quat_to_euler(quat_traj[3-offset:7-offset])
        euler_traj[6-offset:          ] = quat_traj[7-offset:]
    else:
        euler_traj = np.zeros((quat_traj.shape[0] - 1, quat_traj.shape[1]))
        euler_traj[0       :3-offset, :] = quat_traj[0:3-offset, :]
        euler_traj[3-offset:6-offset, :] = quat_to_euler(quat_traj[3-offset:7-offset, :])
        euler_traj[6-offset:        , :] = quat_traj[7-offset:, :]

    return euler_traj