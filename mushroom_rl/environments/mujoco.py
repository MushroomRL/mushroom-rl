import mujoco_py
from mujoco_py import functions as mj_fun
import numpy as np
from enum import Enum
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
import glfw


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


class MuJoCo(Environment):
    """
    Class to create a Mushroom environment using the MuJoCo simulator.

    """
    def __init__(self, file_name, actuation_spec, observation_spec, gamma,
                 horizon, n_substeps=1, n_intermediate_steps=1, additional_data_spec=None,
                 collision_groups=None):
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
            n_substeps (int): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly;
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
        """
        # Create the simulation
        self._sim = self._load_simulation(file_name, n_substeps)

        self._n_intermediate_steps = n_intermediate_steps
        self._viewer = None
        self._state = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        if len(actuation_spec) == 0:
            self._action_indices = [i for i in range(0, len(self._sim.model._actuator_name2id))]
        else:
            self._action_indices = []
            for name in actuation_spec:
                self._action_indices.append(self._sim.model._actuator_name2id[name])

        low = []
        high = []
        for index in self._action_indices:
            if self._sim.model.actuator_ctrllimited[index]:
                low.append(self._sim.model.actuator_ctrlrange[index][0])
                high.append(self._sim.model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)
        action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        if len(observation_spec) == 0:
            raise AttributeError("No Environment observations were specified. "
                                 "Add at least one observation to the observation_spec.")
        else:
            self._observation_map = observation_spec

        # We can only specify limits for the joint positions, all other
        # information can be potentially unbounded
        low = []
        high = []
        for name, ot in self._observation_map:
            obs_count = len(self._get_observation(name, ot))
            if obs_count == 1 and ot == ObservationType.JOINT_POS:
                joint_range = self._sim.model.jnt_range[self._sim.model._joint_name2id[name]]
                if not(joint_range[0] == joint_range[1] == 0.0):
                    low.append(joint_range[0])
                    high.append(joint_range[1])
                else:
                    low.extend(-np.inf)
                    high.extend(np.inf)
            else:
                low.extend([-np.inf] * obs_count)
                high.extend([np.inf] * obs_count)
        observation_space = Box(np.array(low), np.array(high))

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
                self.collision_groups[name] = {self._sim.model._geom_name2id[geom_name]
                                               for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        mj_fun.mj_resetData(self._sim.model, self._sim.data)
        self._setup()

        self._state = self._create_observation()
        return self._state

    def step(self, action):
        cur_obs = self._state

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(action)
            self._sim.data.ctrl[self._action_indices] = ctrl_action

            self._simulation_pre_step()

            self._sim.step()

            self._simulation_post_step()

        self._state = self._create_observation()

        self._step_finalize()

        reward = self._reward(cur_obs, action, self._state)

        return self._state, reward, self._is_absorbing(self._state), {}

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)

        self._viewer.render()

    def stop(self):
        if self._viewer is not None:
            v = self._viewer
            self._viewer = None
            glfw.destroy_window(v.window)
            del v

    def _get_observation(self, name, otype):
        if otype == ObservationType.BODY_POS:
            data = self._sim.data.get_body_xpos(name)
        elif otype == ObservationType.BODY_VEL:
            data = self._sim.data.get_body_xvelp(name)
        elif otype == ObservationType.JOINT_POS:
            data = self._sim.data.get_joint_qpos(name)
        elif otype == ObservationType.JOINT_VEL:
            data = self._sim.data.get_joint_qvel(name)
        elif otype == ObservationType.SITE_POS:
            data = self._sim.data.get_site_xpos(name)
        elif otype == ObservationType.SITE_VEL:
            data = self._sim.data.get_site_xvelp(name)
        else:
            raise ValueError('Invalid observation type')

        if hasattr(data, "__len__"):
            return data
        else:
            return [data]

    def _create_observation(self):
        data_obs = [self._get_observation(name, ot)
                    for name, ot in self._observation_map]
        return np.concatenate(data_obs)

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

    def _compute_action(self, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            action (np.ndarray): numpy array with the actions
                provided at every step.

        Returns:
            The action to be set in the actual mujoco simulation.

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
        return np.array(self._get_observation(data_id, otype))

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
            self._sim.data.set_joint_qpos(data_id, value)
        elif otype == ObservationType.JOINT_VEL:
            self._sim.data.set_joint_qvel(data_id, value)
        else:
            data_buffer = self._get_observation(data_id, otype)
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

        for coni in range(0, self._sim.data.ncon):
            con = self._sim.data.contact[coni]

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
        for con_i in range(0, self._sim.data.ncon):
            con = self._sim.data.contact[con_i]

            if (con.geom1 in ids1 and con.geom2 in ids2 or
               con.geom1 in ids2 and con.geom2 in ids1):

                mujoco_py.functions.mj_contactForce(self._sim.model, self._sim.data,
                                                    con_i, c_array)
                return c_array

        return c_array

    def _reward(self, state, action, next_state):
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

    def _is_absorbing(self, state):
        """
        Check whether the given state is an absorbing state or not.

        Args:
            state (np.array): the state of the system.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        raise NotImplementedError

    def _setup(self):
        """
        A function that allows to execute setup code after an environment
        reset.

        """
        pass

    def _load_simulation(self, file_name, n_substeps):
        """
        Load mujoco model. Can be overridden to provide custom load functions.

        Args:
             file_name (string): The path to the XML file with which the
                environment should be created;
            n_substeps (int): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly.

        Returns:
            The loaded mujoco model.

        """
        return mujoco_py.MjSim(mujoco_py.load_model_from_path(file_name),
                               nsubsteps=n_substeps)

    @property
    def _dt(self):
        return self._sim.model.opt.timestep
