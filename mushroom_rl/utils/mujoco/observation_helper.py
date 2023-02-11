import numpy as np
from enum import Enum


class ObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint-/Body-/Site- positions, rotations, and velocities.
    The Observation have the following returns:
        BODY_POS: (3,) x, y, z position of the body
        BODY_ROT: (4,) quaternion of the body
        BODY_VEL: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z
        JOINT_POS: (1,) rotation of the joint OR (7,) position, quaternion of a free joint
        JOINT_VEL: (1,) velocity of the joint OR (6,) FIRST linear then angular velocity !different to BODY_VEL!
        SITE_POS: (3,) x, y, z position of the body
        SITE_ROT: (9,) rotation matrix of the site
    """
    __order__ = "BODY_POS BODY_ROT BODY_VEL JOINT_POS JOINT_VEL SITE_POS SITE_ROT"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_VEL = 2
    JOINT_POS = 3
    JOINT_VEL = 4
    SITE_POS = 5
    SITE_ROT = 6


class ObservationHelper:
    def __init__(self, observation_spec, model, data, max_joint_velocity):
        if len(observation_spec) == 0:
            raise AttributeError("No Environment observations were specified. "
                                 "Add at least one observation to the observation_spec.")

        self.obs_low = []
        self.obs_high = []
        self.joint_pos_idx = []
        self.joint_vel_idx = []
        self.joint_mujoco_idx = []

        self.obs_idx_map = {}

        self.build_omit_idx = {}

        self.observation_spec = observation_spec

        if max_joint_velocity is not None:
            max_joint_velocity = iter(max_joint_velocity)

        current_idx = 0
        for key, name, ot in observation_spec:
            assert key not in self.obs_idx_map.keys(), "Found duplicate key in observation specification: \"%s\"" % key
            obs_count = len(self.get_state(data, name, ot))
            self.obs_idx_map[key] = list(range(current_idx, current_idx + obs_count))
            self.build_omit_idx[key] = []
            if obs_count == 1 and ot == ObservationType.JOINT_POS:
                self.joint_pos_idx.append(current_idx)
                self.joint_mujoco_idx.append(model.joint(name).id)
                if model.joint(name).limited:
                    self.obs_low.append(model.joint(name).range[0])
                    self.obs_high.append(model.joint(name).range[1])
                else:
                    self.obs_low.append(-np.inf)
                    self.obs_high.append(np.inf)

            elif obs_count == 1 and ot == ObservationType.JOINT_VEL:
                self.joint_vel_idx.append(current_idx)
                if max_joint_velocity is None:
                    max_vel = np.inf
                else:
                    max_vel = next(max_joint_velocity)

                self.obs_low.append(-max_vel)
                self.obs_high.append(max_vel)
            else:
                self.obs_low.extend([-np.inf] * obs_count)
                self.obs_high.extend([np.inf] * obs_count)

            current_idx += obs_count

        self.obs_low = np.array(self.obs_low)
        self.obs_high = np.array(self.obs_high)

    def remove_obs(self, key, index):
        """
        Remove an index from the observation. Cannot remove a whole observation, to achieve this just move the
        observation to additional data.
        Helpful for example to remove the z-coordinate from positions if it's not needed
        The index is always of the original observation!
        """
        indices = self.obs_idx_map[key]
        adjusted_index = index - len(self.build_omit_idx[key])

        self.obs_low = np.delete(self.obs_low, indices[adjusted_index])
        self.obs_high = np.delete(self.obs_high, indices[adjusted_index])
        cutoff = indices.pop(adjusted_index)

        for obs_list in self.obs_idx_map.values():
            for idx in range(len(obs_list)):
                if obs_list[idx] > cutoff:
                    obs_list[idx] -= 1

        for i in range(len(self.joint_pos_idx)):
            if self.joint_pos_idx[i] > cutoff:
                self.joint_pos_idx[i] -= 1

        for i in range(len(self.joint_vel_idx)):
            if self.joint_vel_idx[i] > cutoff:
                self.joint_vel_idx[i] -= 1

        self.build_omit_idx[key].append(index)

    def add_obs(self, key, length, min_value=-np.inf, max_value=np.inf):
        """
        Adds an observation entry to the handling logic of the Helper. The observation still has to be manually
        appended to the original observation via _create_observation(self, state), but can get be accessed via
        get_from_obs(self, obs, name, o_type) and is in obs_low / obs_high
        """
        self.obs_idx_map[key] = list(range(len(self.obs_low), len(self.obs_low) + length))

        if hasattr(min_value, "__len__"):
            self.obs_low = np.append(self.obs_low, min_value)
        else:
            self.obs_low = np.append(self.obs_low, [min_value] * length)

        if hasattr(max_value, "__len__"):
            self.obs_high = np.append(self.obs_high, max_value)
        else:
            self.obs_high = np.append(self.obs_high, [max_value] * length)

    def get_from_obs(self, obs, key):
        # Cannot use advanced indexing because it returns a copy.....
        # We want this data to be writeable
        return obs[self.obs_idx_map[key][0]:self.obs_idx_map[key][-1] + 1]

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

    def _build_obs(self, data):
        """
        Builds the observation given the true state of the simulation. The ObservationType documentation
        describes the different returns in detail
        Args:
            data: The data of the mujoco sim

        Returns: np.array with all the observations defined by observation_spec
        """
        observations = []
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            obs = self.get_state(data, name, o_type)
            if len(omit) != 0:
                obs = np.delete(obs, omit)
            observations.append(obs)
        return np.concatenate(observations)

    def _modify_data(self, data, obs):
        """
        Write the values of the observation into the provided mujoco data object. ONLY joint_pos / joint_vel
        observations will have an effect on the simulation when overwritten. Everything else is just discarded by mujoco
        """
        current_idx = 0
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            current_obs = self.get_state(data, name, o_type)
            for i in range(len(current_obs)):
                if i not in omit:
                    current_obs[i] = obs[current_idx]
                    current_idx += 1

    def get_state(self, data, name, o_type):
        """
        Get a single observation from data, given it's name and observation type. The ObservationType documentation
        describes the different returns in detail
        """
        if o_type == ObservationType.BODY_POS:
            obs = data.body(name).xpos
        elif o_type == ObservationType.BODY_ROT:
            obs = data.body(name).xquat
        elif o_type == ObservationType.BODY_VEL:
            obs = data.body(name).cvel
        elif o_type == ObservationType.JOINT_POS:
            obs = data.joint(name).qpos
        elif o_type == ObservationType.JOINT_VEL:
            obs = data.joint(name).qvel
        elif o_type == ObservationType.SITE_POS:
            obs = data.site(name).xpos
        elif o_type == ObservationType.SITE_ROT:
            # Sites don't have rotation quaternion for some reason...
            # x_mat is rotation matrix with shape (9,)
            obs = data.site(name).xmat
        else:
            raise ValueError('Invalid observation type')

        return np.atleast_1d(obs)

    def get_all_observation_keys(self):
        return list(self.obs_idx_map.keys())
