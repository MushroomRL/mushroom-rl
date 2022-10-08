import mujoco
import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.mujoco import *


class MuJoCo(Environment):
    """
    Class to create a Mushroom environment using the MuJoCo simulator.

    """
    def __init__(self, file_name, actuation_spec, observation_spec, gamma,
                 horizon, timestep=1 / 240., n_intermediate_steps=1, additional_data_spec=None,
                 collision_groups=None, **viewer_params):
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
                their type (ObservationType). An entry in the list is given by:
                (name, type);
            gamma (float): The discounting factor of the environment;
            horizon (int): The maximum horizon for the environment;
            timestep (float, 0.00416666666): The timestep used by the MuJoCo
                simulator;
            n_intermediate_steps (int): The number of steps between every action
                taken by the agent. Similar to n_substeps but allows the user
                to modify, control and access intermediate states.
            additional_data_spec (list): A list containing the data fields of
                interest, which should be read from or written to during
                simulation. The entries are given as the following tuples:
                (key, name, type) key is a string for later referencing in the
                "read_data" and "write_data" methods. The name is the name of
                the object in the XML specification and the type is the
                ObservationType;
            collision_groups (list): A list containing groups of geoms for
                which collisions should be checked during simulation via
                ``check_collision``. The entries are given as:
                ``(key, geom_names)``, where key is a string for later
                referencing in the "check_collision" method, and geom_names is
                a list of geom names in the XML specification.
            **viewer_params: other parameters to be passed to the viewer.
                See MujocoGlfwViewer documentation for the available options.
        """
        # Create the simulation
        self.model = mujoco.MjModel.from_xml_path(file_name)
        self.model.opt.timestep = timestep

        self.data = mujoco.MjData(self.model)

        self._n_intermediate_steps = n_intermediate_steps
        self._timestep = timestep
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        if len(actuation_spec) == 0:
            self._action_indices = [i for i in range(0, len(self.data.actuator_force))]
        else:
            self._action_indices = []
            for name in actuation_spec:
                self._action_indices.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
                # self._action_indices.append(self.model.actuator(name).id) Will work in future release of mujoco...

        low = []
        high = []
        for index in self._action_indices:
            if self.model.actuator_ctrllimited[index]:
                low.append(self.model.actuator_ctrlrange[index][0])
                high.append(self.model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)
        action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        self.obs_helper = ObservationHelper(observation_spec, self.model, self.data, max_joint_velocity=3)

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
                self.collision_groups[name] = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                                               for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        mujoco.mj_resetData(self.model, self.data)
        self.setup()

        self._obs = self.obs_helper.build_obs(self.data)
        return self._obs

    def step(self, action):
        cur_obs = self._obs.copy()

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(cur_obs, action)
            self.data.ctrl[self._action_indices] = ctrl_action

            self._simulation_pre_step()

            mujoco.mj_step(self.model, self.data)

            self._simulation_post_step()

        self._obs = self._obs = self.obs_helper.build_obs(self.data)

        self._step_finalize()

        reward = self.reward(cur_obs, action, self._obs)

        return self._obs, reward, self.is_absorbing(self._obs), {}

    def render(self):
        if self._viewer is None:
            self._viewer = MujocoGlfwViewer(self.model, self.dt, **self._viewer_params)

        self._viewer.render(self.data)

    def stop(self):
        if self._viewer is not None:
            v = self._viewer
            self._viewer = None
            del v

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

    def _step_init(self, state, action):
        """
        Allows information to be initialized at the start of a step.
        """
        pass

    def _compute_action(self, state, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            state (np.ndarray): numpy array with the current state of teh simulation;
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
        return np.array(self.obs_helper.get_state(self.data, data_id, otype))

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
            self.data.joint(data_id).qpos = value
        elif otype == ObservationType.JOINT_VEL:
            self.data.joint(data_id).qvel = value
        else:
            data_buffer = self.obs_helper.get_state(self.data, data_id, otype)
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

        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]

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
        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]

            if (con.geom1 in ids1 and con.geom2 in ids2 or
               con.geom1 in ids2 and con.geom2 in ids1):

                mujoco.mj_contactForce(self.model, self.data,
                                                    con_i, c_array)
                return c_array

        return c_array

    def reward(self, state, action, next_state):
        """
        Compute the reward based on the given transition.

        Args:
            state (np.array): the current state of the system;
            action (np.array): the action that is applied in the current state;
            next_state (np.array): the state reached after applying the given
                action.

        Returns:
            The reward as a floating point scalar value.

        """
        raise NotImplementedError

    def is_absorbing(self, state):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            state (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        raise NotImplementedError

    def setup(self):
        """
        A function that allows to execute setup code after an environment
        reset.

        """
        pass

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps
