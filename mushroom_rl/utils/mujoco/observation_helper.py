import mujoco

import numpy as np
from enum import Enum


class ObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint-/Body-/Site- positions and velocities.

    """
    __order__ = "BODY_POS BODY_VEL JOINT_POS JOINT_VEL SITE_POS SITE_VEL"
    BODY_POS = 0
    BODY_VEL = 1
    JOINT_POS = 2
    JOINT_VEL = 3
    SITE_POS = 4
    SITE_VEL = 5


class ObservationHelper:
    def __init__(self, observation_spec, model, data, max_joint_velocity=3):
        if len(observation_spec) == 0:
            raise AttributeError("No Environment observations were specified. "
                                 "Add at least one observation to the observation_spec.")

        self.obs_low = []
        self.obs_high = []
        self.joint_pos_idx = []
        self.joint_vel_idx = []

        self.obs_idx_map = {}

        self.observation_spec = observation_spec
        current_idx = 0
        for name, ot in observation_spec:
            obs_count = len(self.get_state(data, name, ot))
            self.obs_idx_map[(name, ot)] = list(range(current_idx, current_idx + obs_count))
            if obs_count == 1 and ot == ObservationType.JOINT_POS:
                self.joint_pos_idx.append(current_idx)
                if model.joint(name).limited:
                    self.obs_low.append(model.joint(name).range[0])
                    self.obs_high.append(model.joint(name).range[1])
                else:
                    self.obs_low.append(-np.inf)
                    self.obs_high.append(np.inf)

            elif obs_count == 1 and ot == ObservationType.JOINT_VEL:
                self.joint_vel_idx.append(current_idx)
                self.obs_low.append(-max_joint_velocity)
                self.obs_high.append(max_joint_velocity)
            else:
                self.obs_low.extend([-np.inf] * obs_count)
                self.obs_high.extend([np.inf] * obs_count)

            current_idx += obs_count

        self.obs_low = np.array(self.obs_low)
        self.obs_high = np.array(self.obs_high)

    def get_from_obs(self, obs, name, o_type):
        return obs[self.obs_idx_map[(name, o_type)]]

    def get_joint_pos_from_obs(self, obs):
        return obs[self.joint_pos_idx]

    def get_joint_vel_from_obs(self, obs):
        return obs[self.joint_vel_idx]

    def get_obs_limits(self):
        return self.obs_low, self.obs_high

    def get_joint_pos_limits(self):
        return self.obs_low[self.joint_pos_idx], self.obs_high[self.joint_pos_idx]

    def get_joint_vel_limits(self):
        return self.obs_low[self.joint_vel_idx], self.obs_high[self.joint_vel_idx]

    def build_obs(self, data):
        return np.concatenate([self.get_state(data, name, o_type) for name, o_type in self.observation_spec])

    def get_state(self, data, name, o_type):
        if o_type == ObservationType.BODY_POS:
            obs = data.body(name).xpos
        elif o_type == ObservationType.BODY_VEL:
            obs = data.body(name).cvel
        elif o_type == ObservationType.JOINT_POS:
            obs = data.joint(name).qpos
        elif o_type == ObservationType.JOINT_VEL:
            obs = data.joint(name).qvel
        elif o_type == ObservationType.SITE_POS:
            obs = data.site(name).xpos
        elif o_type == ObservationType.SITE_VEL:
            obs = data.site(name).cvel
        else:
            raise ValueError('Invalid observation type')

        return np.atleast_1d(obs)


