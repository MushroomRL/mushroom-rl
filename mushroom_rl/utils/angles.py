# *********************************************************************
# Software License Agreement (BSD License)
#
#  Copyright (c) 2015, Bossa Nova Robotics
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Bossa Nova Robotics nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
# ********************************************************************/

from math import fmod
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_angle_positive(angle):
    """
    Wrap the angle between 0 and 2 * pi.

    Args:
        angle (float): angle to wrap.

    Returns:
         The wrapped angle.

    """
    pi_2 = 2. * np.pi

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
        a -= 2. * np.pi

    return a


def shortest_angular_distance(from_angle, to_angle):
    """
    Compute the shortest distance between two angles

    Args:
        from_angle (float): starting angle;
        to_angle (float): final angle.

    Returns:
        The shortest distance between from_angle and to_angle.

    """
    return normalize_angle(to_angle - from_angle)


def quat_to_euler(quat):
    """
    Convert a quaternion to euler angles.

    Args:
        quat (np.ndarray):  quaternion to be converted, must be in format [w, x, y, z]

    Returns:
        The euler angles [x, y, z] representation of the quaternion

    """
    if len(quat.shape) < 2:
        return R.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')
    else:
        return R.from_quat(quat[[1, 2, 3, 0], :].T).as_euler('xyz').T


def euler_to_quat(euler):
    """
    Convert euler angles into a quaternion.

    Args:
        euler (np.ndarray):  euler angles to be converted

    Returns:
        Quaternion in format [w, x, y, z]

    """
    if len(euler.shape) < 2:
        return R.from_euler('xyz', euler).as_quat()[[3, 0, 1, 2]]
    else:
        return R.from_euler('xyz', euler.T).as_quat()[:, [3, 0, 1, 2]].T
