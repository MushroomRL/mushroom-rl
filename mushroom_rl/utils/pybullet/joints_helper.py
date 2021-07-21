import numpy as np

from .observation import PyBulletObservationType


class JointsHelper(object):
    def __init__(self, client, indexer, observation_spec):
        self._joint_pos_indexes = list()
        self._joint_velocity_indexes = list()
        joint_limits_low = list()
        joint_limits_high = list()
        joint_velocity_limits = list()
        for joint_name, obs_type in observation_spec:
            joint_idx = indexer.get_index(joint_name, obs_type)
            if obs_type == PyBulletObservationType.JOINT_VEL:
                self._joint_velocity_indexes.append(joint_idx[0])

                model_id, joint_id = indexer.joint_map[joint_name]
                joint_info = client.getJointInfo(model_id, joint_id)
                joint_velocity_limits.append(joint_info[11])

            elif obs_type == PyBulletObservationType.JOINT_POS:
                self._joint_pos_indexes.append(joint_idx[0])

                model_id, joint_id = indexer.joint_map[joint_name]
                joint_info = client.getJointInfo(model_id, joint_id)
                joint_limits_low.append(joint_info[8])
                joint_limits_high.append(joint_info[9])

        self._joint_limits_low = np.array(joint_limits_low)
        self._joint_limits_high = np.array(joint_limits_high)
        self._joint_velocity_limits = np.array(joint_velocity_limits)

    def positions(self, state):
        return state[self._joint_pos_indexes]

    def velocities(self, state):
        return state[self._joint_velocity_indexes]

    def limits(self):
        return self._joint_limits_low, self._joint_limits_high

    def velocity_limits(self):
        return self._joint_velocity_limits
