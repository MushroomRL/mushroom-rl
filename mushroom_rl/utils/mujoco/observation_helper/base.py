import numpy as np
from enum import Enum

import mujoco


class ObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint/Body/Site positions, rotations, and velocities.
    The Observation have the following returns::

        BODY_POS: (3,) x, y, z position of the body
        BODY_ROT: (4,) quaternion of the body
        BODY_VEL: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z, in local frame
        BODY_VEL_WORLD: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z, in world frame
        JOINT_POS: (1,) rotation of the joint OR (7,) position, quaternion of a free joint
        JOINT_VEL: (1,) velocity of the joint OR (6,) FIRST linear then angular velocity !different to BODY_VEL!
        SITE_POS: (3,) x, y, z position of the body
        SITE_ROT: (9,) rotation matrix of the site
    """

    __order__ = "BODY_POS BODY_ROT BODY_VEL BODY_VEL_WORLD JOINT_POS JOINT_VEL SITE_POS SITE_ROT"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_VEL = 2
    BODY_VEL_WORLD = 3
    JOINT_POS = 4
    JOINT_VEL = 5
    SITE_POS = 6
    SITE_ROT = 7


class ObservationHelper:
    """
    Base class for the observation helpers of the MuJoCo environments. It maps an observation specification onto the
    flat observation vector, gives access to the entries of that vector and to their limits, and declares the interface
    a simulation backend has to implement to read observations out of a simulation and write them back into it.

    """
    _FIXED_OBS_SIZE = {
        ObservationType.BODY_POS: 3,
        ObservationType.BODY_ROT: 4,
        ObservationType.BODY_VEL: 6,
        ObservationType.BODY_VEL_WORLD: 6,
        ObservationType.SITE_POS: 3,
        ObservationType.SITE_ROT: 9,
    }

    def __init__(self, observation_spec, model, max_joint_velocity):
        """
        Constructor.

        Args:
            observation_spec (list): the observation specification, as a list of (key, name, ObservationType) tuples;
            model: the MuJoCo model the observations are read from;
            max_joint_velocity (list, None): the maximum velocity of every JOINT_VEL entry of the specification, in
                the order the entries appear, or None to leave them unbounded.

        """
        if len(observation_spec) == 0:
            raise AttributeError(
                "No Environment observations were specified. "
                "Add at least one observation to the observation_spec."
            )

        self.obs_low = []
        self.obs_high = []
        self.joint_pos_idx = []
        self.joint_vel_idx = []
        self.joint_mujoco_idx = []

        self.obs_idx_map = {}

        self.build_omit_idx = {}

        self.observation_spec = observation_spec

        self._model = model
        self._size_cache = {}
        self._address_cache = {}

        if max_joint_velocity is not None:
            max_joint_velocity = iter(max_joint_velocity)

        current_idx = 0
        for key, name, ot in observation_spec:
            assert key not in self.obs_idx_map.keys(), (
                'Found duplicate key in observation specification: "%s"' % key
            )
            obs_count = self._obs_size(name, ot)
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

    def build_obs(self):
        """
        Build the observation vector from the current state of the simulation.

        Returns:
            The observation defined by the observation specification.

        """
        raise NotImplementedError

    def modify_data(self, obs, env_indices=None):
        """
        Write the values of the given observation into the simulation. Only JOINT_POS and JOINT_VEL entries have an
        effect on the simulation; everything else is discarded by MuJoCo.

        Args:
            obs: the observation to write;
            env_indices (None): the worlds to write to, or None to write to all of them.

        """
        raise NotImplementedError

    def get_state(self, name, o_type, env_indices=None):
        """
        Read a single entry from the simulation, given its name and observation type.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to read;
            env_indices (None): the worlds to read, or None to read all of them.

        Returns:
            The requested data.

        """
        raise NotImplementedError

    def set_state(self, name, o_type, value, env_indices=None):
        """
        Write a single entry into the simulation, given its name and observation type.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to write;
            value: the data to write;
            env_indices (None): the worlds to write to, or None to write to all of them.

        """
        raise NotImplementedError

    def remove_obs(self, key, index):
        """
        Drop one entry of an observation from the observation vector, shortening it and shifting down every index
        after it. A whole observation cannot be dropped; move it to the additional data instead.

        Args:
            key (str): the key of the observation the entry belongs to;
            index (int): the position of the entry inside that observation, always counted on the original
                observation.

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
        Append an observation to the end of the observation vector, so that it can be read by key and is covered by
        the limits. Its values are not produced by the helper: the environment has to append them to the observation
        it builds.

        Args:
            key (str): the key the observation is read by;
            length (int): the number of values of the observation;
            min_value (float, Array, -np.inf): the lower limit of the observation, either one value for all of it or
                one value per entry;
            max_value (float, Array, np.inf): the upper limit of the observation, either one value for all of it or
                one value per entry.

        """
        self.obs_idx_map[key] = list(
            range(len(self.obs_low), len(self.obs_low) + length)
        )

        if hasattr(min_value, "__len__"):
            self.obs_low = np.append(self.obs_low, min_value)
        else:
            self.obs_low = np.append(self.obs_low, [min_value] * length)

        if hasattr(max_value, "__len__"):
            self.obs_high = np.append(self.obs_high, max_value)
        else:
            self.obs_high = np.append(self.obs_high, [max_value] * length)

    def get_from_obs(self, obs, key):
        """
        Single out one observation from a given observation vector.

        Args:
            obs (Array): the observation vector to read, with the observation dimension last;
            key (str): the key of the observation to single out.

        Returns:
            The values of the named observation, as a view of the given vector: writing into the result writes into
            the vector.

        """
        return obs[..., self.obs_idx_map[key][0]:self.obs_idx_map[key][-1] + 1]

    def get_joint_pos_from_obs(self, obs):
        """
        Args:
            obs (Array): the observation vector to read.

        Returns:
            The values of every one-dimensional JOINT_POS observation, in the order they appear in the vector.

        """
        return obs[self.joint_pos_idx]

    def get_joint_vel_from_obs(self, obs):
        """
        Args:
            obs (Array): the observation vector to read.

        Returns:
            The values of every one-dimensional JOINT_VEL observation, in the order they appear in the vector.

        """
        return obs[self.joint_vel_idx]

    def get_obs_limits(self):
        """
        Returns:
            The lower and the upper limit of every entry of the observation vector.

        """
        return self.obs_low, self.obs_high

    def get_joint_pos_limits(self):
        """
        Returns:
            The lower and the upper limit of every one-dimensional JOINT_POS observation.

        """
        return self.obs_low[self.joint_pos_idx], self.obs_high[self.joint_pos_idx]

    def get_joint_vel_limits(self):
        """
        Returns:
            The lower and the upper limit of every one-dimensional JOINT_VEL observation.

        """
        return self.obs_low[self.joint_vel_idx], self.obs_high[self.joint_vel_idx]

    def get_all_observation_keys(self):
        """
        Returns:
            The key of every observation, in the order the observations appear in the observation vector.

        """
        return list(self.obs_idx_map.keys())

    def _address(self, name, ot):
        """
        Return where an observation entry sits in the raw data field of its type: the index of the body or of the
        site, or the range of values of the joint.
        """
        address = self._address_cache.get((name, ot))

        if address is None:
            if ot in (ObservationType.BODY_POS, ObservationType.BODY_ROT, ObservationType.BODY_VEL,
                      ObservationType.BODY_VEL_WORLD):
                address = self._model.body(name).id
            elif ot in (ObservationType.SITE_POS, ObservationType.SITE_ROT):
                address = self._model.site(name).id
            elif ot == ObservationType.JOINT_POS:
                adr = self._model.jnt_qposadr[self._model.joint(name).id]
                address = slice(adr, adr + self._obs_size(name, ot))
            elif ot == ObservationType.JOINT_VEL:
                adr = self._model.jnt_dofadr[self._model.joint(name).id]
                address = slice(adr, adr + self._obs_size(name, ot))
            else:
                raise ValueError(f"Invalid observation type: {ot}")

            self._address_cache[(name, ot)] = address

        return address

    def _obs_size(self, name, ot):
        """
        Return the number of scalar values for this observation entry,
        computed from the model alone (no simulation data required).
        """
        size = self._size_cache.get((name, ot))

        if size is None:
            if ot in self._FIXED_OBS_SIZE:
                if ot in (ObservationType.SITE_POS, ObservationType.SITE_ROT):
                    self._model.site(name)
                else:
                    self._model.body(name)
                size = self._FIXED_OBS_SIZE[ot]
            elif ot == ObservationType.JOINT_POS:
                jnt_type = self._model.jnt_type[self._model.joint(name).id]
                size = self._joint_type_size(jnt_type, free_size=7, ball_size=4)
            elif ot == ObservationType.JOINT_VEL:
                jnt_type = self._model.jnt_type[self._model.joint(name).id]
                size = self._joint_type_size(jnt_type, free_size=6, ball_size=3)
            else:
                raise ValueError(f"Invalid observation type: {ot}")

            self._size_cache[(name, ot)] = size

        return size

    @staticmethod
    def _joint_type_size(jnt_type, free_size, ball_size):
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            return free_size
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            return ball_size
        else:
            return 1
