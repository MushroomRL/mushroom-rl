import pybullet
import pybullet_data
import time
import numpy as np
from enum import Enum
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.viewer import ImageViewer


class PyBulletObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint-/Body-/Site- positions and velocities.

    """
    __order__ = "JOINT_POS JOINT_VEL"
    JOINT_POS = 0
    JOINT_VEL = 1


class PyBulletViewer(ImageViewer):
    def __init__(self, size, dt, distance=4, origin=(0,0,1), angles=(0, -45, 60),
                 fov=60, aspect=1, near_val=0.01, far_val=100):
        self._size = size
        self._distance = distance
        self._origin = origin
        self._angles = angles
        self._fov = fov
        self._aspect = aspect
        self._near_val = near_val
        self._far_val = far_val
        super().__init__(size, dt)

    def display(self):
        img = self._get_image()
        super().display(img)

    def _get_image(self):
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self._origin,
                                                                 distance=self._distance,
                                                                 roll=self._angles[0],
                                                                 pitch=self._angles[1],
                                                                 yaw=self._angles[2],
                                                                 upAxisIndex=2)
        proj_matrix = pybullet.computeProjectionMatrixFOV(fov=self._fov, aspect=self._aspect,
                                                          nearVal=self._near_val, farVal=self._far_val)
        (_, _, px, _, _) = pybullet.getCameraImage(width=self._size[0],
                                                   height=self._size[1],
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(np.array(px), (self._size[0], self._size[1], -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class PyBullet(Environment):
    """
    Class to create a Mushroom environment using the PyBullet simulator.

    """
    def __init__(self, file_name, actuation_spec, control_mode, observation_spec, gamma,
                 horizon, timestep=1/240, n_intermediate_steps=1):
        """
        Constructor.

        Args:
            file_name (string): The path to the urdf file with which the
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
            timestep (float, 0.00416666666): The timestep used by the PyBullet
                simulator;
            n_intermediate_steps (int): The number of steps between every action
                taken by the agent. Allows the user
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

        self._timestep = timestep

        # Create the simulation
        pybullet.connect(pybullet.DIRECT)
        pybullet.setTimeStep(self._timestep)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._model_id = pybullet.loadURDF(file_name)

        self.n_intermediate_steps = n_intermediate_steps
        self.viewer = None
        self._state = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        self._joint_map = dict()
        for id in range(pybullet.getNumJoints(self._model_id)):
            name = pybullet.getJointInfo(self._model_id, id)[1].decode('UTF-8')
            self._joint_map[name] = id

        if len(actuation_spec) == 0:
            self.action_indices = [i for i in pybullet.getNumJoints(self._model_id)]
        else:
            self.action_indices = []
            for name in actuation_spec:
                if name in self._joint_map:
                    self.action_indices.append(self._joint_map[name])

        self._control_mode = control_mode

        low = []
        high = []
        for index in self.action_indices:
            #TODO add limits
            low.append(-np.inf)
            high.append(np.inf)
        action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        if len(observation_spec) == 0:
            raise AttributeError("No Environment observations were specified. "
                                 "Add at least one observation to the observation_spec.")
        else:
            self.observation_map = observation_spec

        # We can only specify limits for the joint positions, all other
        # information can be potentially unbounded
        # TODO fix this
        low = np.array([-np.inf]*len(self.observation_map))
        high = np.array([np.inf]*len(self.observation_map))
        observation_space = Box(low, high)


        # Pre-process the additional data to allow easier writing and reading
        # to and from arrays in MuJoCo
        # self.additional_data = {}
        # for key, name, ot in additional_data_spec:
        #     self.additional_data[key] = (name, ot)

        # Pre-process the collision groups for "fast" detection of contacts
        # self.collision_groups = {}
        # if self.collision_groups is not None:
        #     for name, geom_names in collision_groups:
        #         self.collision_groups[name] = {self.sim.model._geom_name2id[geom_name]
        #                                        for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)
        self._viewer = PyBulletViewer((500, 500), self._timestep*self.n_intermediate_steps)
        self._initial_state = pybullet.saveState()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        pybullet.restoreState(self._initial_state)
        #pybullet.resetSimulation()
        self.setup()
        pybullet.stepSimulation()
        self._state = self._create_observation()
        return self._state

    def render(self):
        self._viewer.display()

    def stop(self):
        pass

    def _create_observation(self):
        data_obs = list()

        for name, type in self.observation_map:
            idx = self._joint_map[name]
            pos, vel, _, _ = pybullet.getJointState(self._model_id, idx)
            if type == PyBulletObservationType.JOINT_POS:
                data_obs.append(pos)
            elif type == PyBulletObservationType.JOINT_VEL:
                data_obs.append(vel)

        return np.array(data_obs)

    def step(self, action):
        cur_obs = self._state

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self.n_intermediate_steps):

            ctrl_action = self._compute_action(action)
            self._apply_control(ctrl_action)

            self._simulation_pre_step()

            pybullet.stepSimulation()

            self._simulation_post_step()

        self._state = self._create_observation()

        self._step_finalize()

        reward = self.reward(cur_obs, action, self._state)

        return self._state, reward, self.is_absorbing(self._state), {}

    def _apply_control(self, action):
        if self._control_mode is pybullet.POSITION_CONTROL:
            kwargs = dict(targetPositions=action)
        elif self._control_mode is pybullet.VELOCITY_CONTROL:
            kwargs = dict(targetVelocities=action)
        elif self._control_mode is pybullet.TORQUE_CONTROL:
            kwargs = dict(forces=action)
        else:
            raise NotImplementedError

        pybullet.setJointMotorControlArray(self._model_id, self.action_indices,
                                           self._control_mode, **kwargs)

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
