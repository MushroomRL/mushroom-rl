import numpy as np

import mujoco

from mushroom_rl.utils.mujoco.observation_helper.base import ObservationHelper, ObservationType


class MuJoCoObservationHelper(ObservationHelper):
    """
    Observation helper of a standard MuJoCo simulation, reading from and writing to a single MjData structure.

    """
    def __init__(self, observation_spec, model, data, max_joint_velocity):
        """
        Constructor.

        Args:
            observation_spec (list): the observation specification, as a list of (key, name, ObservationType) tuples;
            model: the MuJoCo model the observations are read from;
            data: the MuJoCo data structure the observations are read from and written to;
            max_joint_velocity (list, None): the maximum velocity of every JOINT_VEL entry of the specification, in
                the order the entries appear, or None to leave them unbounded.

        """
        super().__init__(observation_spec, model, max_joint_velocity)

        self._data = data

    def build_obs(self):
        """
        Build the observation vector from the current state of the simulation. The ObservationType documentation
        describes the different returns in detail.

        Returns:
            A np.array with all the observations defined by the observation specification.

        """
        observations = []
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            obs = self.get_state(name, o_type)
            if len(omit) != 0:
                obs = np.delete(obs, omit)
            observations.append(obs)
        return np.concatenate(observations)

    def modify_data(self, obs, env_indices=None):
        """
        Write the values of the observation into the MuJoCo data structure. ONLY joint_pos / joint_vel observations
        will have an effect on the simulation when overwritten. Everything else is just discarded by mujoco.

        Args:
            obs (np.array): the observation to write;
            env_indices (None): unused, a standard MuJoCo simulation holds a single world.

        """
        assert env_indices is None, "A standard MuJoCo simulation holds a single world."

        current_idx = 0
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            current_obs = self.get_state(name, o_type)
            for i in range(len(current_obs)):
                if i not in omit:
                    current_obs[i] = obs[current_idx]
                    current_idx += 1

    def get_state(self, name, o_type, env_indices=None):
        """
        Get a single observation from the data, given its name and observation type. The ObservationType documentation
        describes the different returns in detail.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to read;
            env_indices (None): unused, a standard MuJoCo simulation holds a single world.

        Returns:
            The requested data, as a one-dimensional np.array.

        """
        assert env_indices is None, "A standard MuJoCo simulation holds a single world."

        if o_type == ObservationType.BODY_POS:
            obs = self._data.xpos[self._address(name, o_type)]
        elif o_type == ObservationType.BODY_ROT:
            obs = self._data.xquat[self._address(name, o_type)]
        elif (
            o_type == ObservationType.BODY_VEL
            or o_type == ObservationType.BODY_VEL_WORLD
        ):
            local = o_type == ObservationType.BODY_VEL
            obs = np.empty(6)
            mujoco.mj_objectVelocity(
                self._model, self._data, mujoco.mjtObj.mjOBJ_XBODY, self._address(name, o_type), obs, local
            )
        elif o_type == ObservationType.JOINT_POS:
            obs = self._data.qpos[self._address(name, o_type)]
        elif o_type == ObservationType.JOINT_VEL:
            obs = self._data.qvel[self._address(name, o_type)]
        elif o_type == ObservationType.SITE_POS:
            obs = self._data.site_xpos[self._address(name, o_type)]
        elif o_type == ObservationType.SITE_ROT:
            obs = self._data.site_xmat[self._address(name, o_type)]
        else:
            raise ValueError("Invalid observation type")

        return np.atleast_1d(obs)

    def set_state(self, name, o_type, value, env_indices=None):
        """
        Write a single entry into the data, given its name and observation type.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to write;
            value (np.array): the data to write;
            env_indices (None): unused, a standard MuJoCo simulation holds a single world.

        """
        assert env_indices is None, "A standard MuJoCo simulation holds a single world."

        if o_type == ObservationType.JOINT_POS:
            self._data.joint(name).qpos = value
        elif o_type == ObservationType.JOINT_VEL:
            self._data.joint(name).qvel = value
        else:
            data_buffer = self.get_state(name, o_type)
            data_buffer[:] = value
