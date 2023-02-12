import mujoco
import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.mujoco import *


class MuJoCo(Environment):
    """
    Class to create a Mushroom environment using the MuJoCo simulator.
    """

    def __init__(self, file_name, actuation_spec, observation_spec, gamma, horizon, timestep=None, n_substeps=1,
                 n_intermediate_steps=1, additional_data_spec=None, collision_groups=None, max_joint_vel=None,
                 **viewer_params):
        """
        Constructor.

        Args:
             file_name (string): The path to the XML file with which the
                environment should be created;
             actuation_spec (list): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used;
             observation_spec (list): A list containing the names of data that
                should be made available to the agent as an observation and
                their type (ObservationType). They are combined with a key,
                which is used to access the data. An entry in the list
                is given by: (key, name, type). The name can later be used
                to retrieve specific observations;
             gamma (float): The discounting factor of the environment;
             horizon (int): The maximum horizon for the environment;
             timestep (float): The timestep used by the MuJoCo
                simulator. If None, the default timestep specified in the XML will be used;
             n_substeps (int, 1): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly;
             n_intermediate_steps (int, 1): The number of steps between every action
                taken by the agent. Similar to n_substeps but allows the user
                to modify, control and access intermediate states.
             additional_data_spec (list, None): A list containing the data fields of
                interest, which should be read from or written to during
                simulation. The entries are given as the following tuples:
                (key, name, type) key is a string for later referencing in the
                "read_data" and "write_data" methods. The name is the name of
                the object in the XML specification and the type is the
                ObservationType;
             collision_groups (list, None): A list containing groups of geoms for
                which collisions should be checked during simulation via
                ``check_collision``. The entries are given as:
                ``(key, geom_names)``, where key is a string for later
                referencing in the "check_collision" method, and geom_names is
                a list of geom names in the XML specification.
             max_joint_vel (list, None): A list with the maximum joint velocities which are provided in the mdp_info.
                The list has to define a maximum velocity for every occurrence of JOINT_VEL in the observation_spec. The
                velocity will not be limited in mujoco
             **viewer_params: other parameters to be passed to the viewer.
                See MujocoGlfwViewer documentation for the available options.

        """
        # Create the simulation
        self._model = mujoco.MjModel.from_xml_path(file_name)
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep

        self._data = mujoco.MjData(self._model)

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        if len(actuation_spec) == 0:
            self._action_indices = [i for i in range(0, len(self._data.actuator_force))]
        else:
            self._action_indices = []
            for name in actuation_spec:
                self._action_indices.append(self._model.actuator(name).id)

        low = []
        high = []
        for index in self._action_indices:
            if self._model.actuator_ctrllimited[index]:
                low.append(self._model.actuator_ctrlrange[index][0])
                high.append(self._model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)
        action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        self.obs_helper = ObservationHelper(observation_spec, self._model, self._data, max_joint_velocity=max_joint_vel)

        observation_space = Box(*self.obs_helper.get_obs_limits())

        # Pre-process the additional data to allow easier writing and reading
        # to and from arrays in MuJoCo
        self.additional_data = {}
        if additional_data_spec is not None:
            for key, name, ot in additional_data_spec:
                self.additional_data[key] = (name, ot)

        # Pre-process the collision groups for "fast" detection of contacts
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                                               for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        mdp_info = self._modify_mdp_info(mdp_info)

        # set the warning callback to stop the simulation when a mujoco warning occurs
        mujoco.set_mju_user_warning(self.user_warning_raise_exception)

        super().__init__(mdp_info)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, obs=None):
        mujoco.mj_resetData(self._model, self._data)
        self.setup(obs)

        self._obs = self._create_observation(self.obs_helper._build_obs(self._data))
        return self._modify_observation(self._obs)

    def step(self, action):
        cur_obs = self._obs.copy()

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(cur_obs, action)
            self._data.ctrl[self._action_indices] = ctrl_action

            self._simulation_pre_step()

            mujoco.mj_step(self._model, self._data, self._n_substeps)

            self._simulation_post_step()

            cur_obs = self._create_observation(self.obs_helper._build_obs(self._data))

        self._step_finalize()

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs

        return self._modify_observation(cur_obs), reward, absorbing, info

    def render(self):
        if self._viewer is None:
            self._viewer = MujocoGlfwViewer(self._model, self.dt, **self._viewer_params)

        self._viewer.render(self._data)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def _modify_mdp_info(self, mdp_info):
        """
        This method can be overridden to modify the automatically generated MDPInfo data structure.
        By default, returns the given mdp_info structure unchanged.

        Args:
            mdp_info (MDPInfo): the MDPInfo structure automatically computed by the environment.

        Returns:
            The modified MDPInfo data structure.

        """
        return mdp_info

    def _create_observation(self, obs):
        """
        This method can be overridden to create a custom observation. Should be used to append observation which have
        been registered via obs_help.add_obs(self, name, o_type, length, min_value, max_value)

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def _create_info_dictionary(self, obs):
        """
        This method can be overridden to create a custom info dictionary.

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The information dictionary.

        """
        return {}

    def _modify_observation(self, obs):
        """
        This method can be overridden to edit the created observation. This is done after the reward and absorbing
        functions are evaluated. Especially useful to transform the observation into different frames. If the original
        observation order is not preserved, the helper functions in ObervationHelper breaks.

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """
        return action

    def _step_init(self, obs, action):
        """
        Allows information to be initialized at the start of a step.

        """
        pass

    def _compute_action(self, obs, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            obs (np.ndarray): numpy array with the current state of teh simulation;
            action (np.ndarray): numpy array with the actions, provided at every step.

        Returns:
            The action to be set in the actual pybullet simulation.

        """
        return action

    def _simulation_pre_step(self):
        """
        Allows information to be accesed and changed at every intermediate step
        before taking a step in the mujoco simulation.
        Can be usefull to apply an external force/torque to the specified bodies.

        ex: apply a force over X to the torso:
        force = [200, 0, 0]
        torque = [0, 0, 0]
        self.sim.data.xfrc_applied[self.sim.model._body_name2id["torso"],:] = force + torque

        """
        pass

    def _simulation_post_step(self):
        """
        Allows information to be accesed at every intermediate step
        after taking a step in the mujoco simulation.
        Can be usefull to average forces over all intermediate steps.

        """
        pass

    def _step_finalize(self):
        """
        Allows information to be accesed at the end of a step.
        """
        pass

    def _read_data(self, name):
        """
        Read data form the MuJoCo data structure.

        Args:
            name (string): A name referring to an entry contained the
                additional_data_spec list handed to the constructor.

        Returns:
            The desired data as a one-dimensional numpy array.

        """
        data_id, otype = self.additional_data[name]
        return np.array(self.obs_helper.get_state(self._data, data_id, otype))

    def _write_data(self, name, value):
        """
        Write data to the MuJoCo data structure.

        Args:
            name (string): A name referring to an entry contained in the
                additional_data_spec list handed to the constructor;
            value (ndarray): The data that should be written.

        """

        data_id, otype = self.additional_data[name]
        if otype == ObservationType.JOINT_POS:
            self._data.joint(data_id).qpos = value
        elif otype == ObservationType.JOINT_VEL:
            self._data.joint(data_id).qvel = value
        else:
            data_buffer = self.obs_helper.get_state(self._data, data_id, otype)
            data_buffer[:] = value

    def _check_collision(self, group1, group2):
        """
        Check for collision between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A flag indicating whether a collision occurred between the given
            groups or not.

        """

        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        for coni in range(0, self._data.ncon):
            con = self._data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False

    def _get_collision_force(self, group1, group2):
        """
        Returns the collision force and torques between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A 6D vector specifying the collision forces/torques[3D force + 3D torque]
            between the given groups. Vector of 0's in case there was no collision.
            http://mujoco.org/book/programming.html#siContact

        """

        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        c_array = np.zeros(6, dtype=np.float64)
        for con_i in range(0, self._data.ncon):
            con = self._data.contact[con_i]

            if (con.geom1 in ids1 and con.geom2 in ids2 or
               con.geom1 in ids2 and con.geom2 in ids1):

                mujoco.mj_contactForce(self._model, self._data,
                                       con_i, c_array)
                return c_array

        return c_array

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward based on the given transition.

        Args:
            obs (np.array): the current state of the system;
            action (np.array): the action that is applied in the current state;
            next_obs (np.array): the state reached after applying the given
                action.
            absorbing (bool): whether next_state is an absorbing state or not.

        Returns:
            The reward as a floating point scalar value.

        """
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            obs (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        raise NotImplementedError

    def setup(self, obs):
        """
        A function that allows to execute setup code after an environment
        reset.

        """
        if obs is not None:
            self.obs_helper._modify_data(self._data, obs)


    def get_all_observation_keys(self):
        """
        A function that returns all observation keys defined in the observation specification.

        Returns:
            A list of observation keys.

        """
        return self.obs_helper.get_all_observation_keys()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps

    @staticmethod
    def user_warning_raise_exception(warning):
        """
        Detects warnings in Mujoco and raises the respective exception.

        Args:
            warning: Mujoco warning.

        """
        if 'Pre-allocated constraint buffer is full' in warning:
            raise RuntimeError(warning + 'Increase njmax in mujoco XML')
        elif 'Pre-allocated contact buffer is full' in warning:
            raise RuntimeError(warning + 'Increase njconmax in mujoco XML')
        elif 'Unknown warning type' in warning:
            raise RuntimeError(warning + 'Check for NaN in simulation.')
        else:
            raise RuntimeError('Got MuJoCo Warning: ' + warning)
