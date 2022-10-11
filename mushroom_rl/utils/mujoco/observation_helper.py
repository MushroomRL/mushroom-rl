import mujoco
import copy
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

        self.build_omit_idx = {}


        self.observation_spec = observation_spec
        current_idx = 0
        for name, ot in observation_spec:
            obs_count = len(self.get_state(data, name, ot))
            self.obs_idx_map[(name, ot)] = list(range(current_idx, current_idx + obs_count))
            self.build_omit_idx[(name, ot)] = []
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

    def remove_obs(self, name, o_type, index):
        """
        Remove an index from the observation. Cannot remove a whole observation, to achieve this just move the
        observation to additional data.
        Helpful for example to remove the z-coordinate from positions if it's not needed
        The index is always of the original observation!
        """
        indices = self.obs_idx_map[(name, o_type)]
        adjusted_index = index - sum([0 if el < index else 1 for el in self.build_omit_idx[(name, o_type)]])

        self.obs_low = np.delete(self.obs_low, indices[adjusted_index])
        self.obs_high = np.delete(self.obs_high, indices[adjusted_index])
        cutoff = indices.pop(adjusted_index)

        for obs_list in self.obs_idx_map.values():
            for idx in range(len(obs_list)):
                if obs_list[idx] > cutoff:
                    obs_list[idx] -= 1

        self.build_omit_idx[(name, o_type)].append(index)

    def add_obs(self, name, o_type, length, min_value=-np.inf, max_value=np.inf):
        """
        Adds an observation entry to the handling logic of the Helper. The observation still has to be manually
        appended to the original observation via _create_observation(self, state), but can get be accessed via
        get_from_obs(self, obs, name, o_type) and is in obs_low / obs_high
        """
        self.obs_idx_map[(name, o_type)] = list(range(len(self.obs_low), len(self.obs_low) + length))

        if hasattr(min_value, "__len__"):
            self.obs_low = np.append(self.obs_low, min_value)
        else:
            self.obs_low = np.append(self.obs_low, [min_value] * length)

        if hasattr(max_value, "__len__"):
            self.obs_high = np.append(self.obs_high, max_value)
        else:
            self.obs_high = np.append(self.obs_high, [max_value] * length)

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
        observations = []
        for name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[(name, o_type)])
            obs = self.get_state(data, name, o_type)
            if len(omit) != 0:
                obs = np.delete(obs, omit)
            observations.append(obs)
        return np.concatenate(observations)

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


