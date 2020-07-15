import pybullet
import pybullet_data
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
    __order__ = "BODY_POS BODY_LIN_VEL BODY_ANG_VEL JOINT_POS JOINT_VEL LINK_POS LINK_LIN_VEL LINK_ANG_VEL"
    BODY_POS = 0
    BODY_LIN_VEL = 1
    BODY_ANG_VEL = 2
    JOINT_POS = 3
    JOINT_VEL = 4
    LINK_POS = 5
    LINK_LIN_VEL = 6
    LINK_ANG_VEL = 7


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
    def __init__(self, files, actuation_spec, observation_spec, gamma,
                 horizon, timestep=1/240, n_intermediate_steps=1, debug_gui=False):
        """
        Constructor.

        Args:
            files (list): Paths to the URDF files to load;
            actuation_spec (list): A list of tuples specifying the names of the
                joints which should be controllable by the agent and tehir control mode.
                 Can be left empty when all actuators should be used in position control;
            observation_spec (list): A list containing the names of data that
                should be made available to the agent as an observation and
                their type (ObservationType). An entry in the list is given by:
                (name, type);
            gamma (float): The discounting factor of the environment;
            horizon (int): The maximum horizon for the environment;
            timestep (float, 0.00416666666): The timestep used by the PyBullet
                simulator;
            n_intermediate_steps (int): The number of steps between every action
                taken by the agent. Allows the user to modify, control and
                access intermediate states;
        """

        # Store simulation parameters
        self._timestep = timestep
        self._n_intermediate_steps = n_intermediate_steps

        # Create the simulation and viewer
        if debug_gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setTimeStep(self._timestep)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._viewer = PyBulletViewer((500, 500), self._timestep * self._n_intermediate_steps)
        self._state = None

        # Load model and create access maps
        self._model_map = dict()
        for file_name, kwargs in files.items():
            model_id = pybullet.loadURDF(file_name, **kwargs)
            model_name = pybullet.getBodyInfo(model_id)[1].decode('UTF-8')
            self._model_map[model_name] = model_id
        self._model_map.update(self._custom_load_models())

        self._joint_map = dict()
        self._link_map = dict()
        for model_id in self._model_map.values():
            for joint_id in range(pybullet.getNumJoints(model_id)):
                joint_data = pybullet.getJointInfo(model_id, joint_id)
                if joint_data[2] != pybullet.JOINT_FIXED:
                    joint_name = joint_data[1].decode('UTF-8')
                    self._joint_map[joint_name] = (model_id, joint_id)
                link_name = joint_data[12].decode('UTF-8')
                self._link_map[link_name] = (model_id, joint_id)

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        assert(len(actuation_spec) > 0)
        self._action_data = list()
        for name, mode in actuation_spec:
            if name in self._joint_map:
                data = self._joint_map[name] + (mode,)
                self._action_data.append(data)

        low, high = self._compute_action_limits()
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
        low, high = self._compute_observation_limits()
        observation_space = Box(low, high)

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

        # Save initial state of the MDP
        self._initial_state = pybullet.saveState()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        pybullet.restoreState(self._initial_state)
        self.setup()
        self._state = self._create_observation()
        return self._state

    def render(self):
        self._viewer.display()

    def stop(self):
        pass

    def step(self, action):
        cur_obs = self._state

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(action)
            self._apply_control(ctrl_action)

            self._simulation_pre_step()

            pybullet.stepSimulation()

            self._simulation_post_step()

        self._state = self._create_observation()

        self._step_finalize()

        reward = self.reward(cur_obs, action, self._state)

        return self._state, reward, self.is_absorbing(self._state), {}

    def _compute_action_limits(self):
        low = list()
        high = list()

        for model_id, joint_id, mode in self._action_data:
            joint_info = pybullet.getJointInfo(model_id, joint_id)
            if mode is pybullet.POSITION_CONTROL:
                low.append(joint_info[8])
                high.append(joint_info[9])
            elif mode is pybullet.VELOCITY_CONTROL:
                low.append(-joint_info[11])
                high.append(joint_info[11])
            elif mode is pybullet.TORQUE_CONTROL:
                low.append(-joint_info[10])
                high.append(joint_info[10])
            else:
                raise NotImplementedError

        return np.array(low), np.array(high)

    def _compute_observation_limits(self):
        low = list()
        high = list()

        for name, obs_type in self._observation_map:
            if obs_type is PyBulletObservationType.BODY_POS \
               or obs_type is PyBulletObservationType.BODY_LIN_VEL \
               or obs_type is PyBulletObservationType.BODY_ANG_VEL:
                n_dim = 7 if obs_type is PyBulletObservationType.BODY_POS else 3
                low += [-np.inf] * n_dim
                high += [-np.inf] * n_dim
            elif obs_type is PyBulletObservationType.LINK_POS \
                 or obs_type is PyBulletObservationType.LINK_LIN_VEL \
                 or obs_type is PyBulletObservationType.LINK_ANG_VEL:
                n_dim = 7 if obs_type is PyBulletObservationType.LINK_POS else 3
                low += [-np.inf] * n_dim
                high += [-np.inf] * n_dim
            else:
                model_id, joint_id = self._joint_map[name]
                joint_info = pybullet.getJointInfo(model_id, joint_id)

                if obs_type is PyBulletObservationType.JOINT_POS:
                    low.append(joint_info[8])
                    high.append(joint_info[9])
                else:
                    low.append(-np.inf)
                    high.append(np.inf)

        return np.array(low), np.array(high)

    def _create_observation(self):
        data_obs = list()

        for name, obs_type in self._observation_map:
            if obs_type is PyBulletObservationType.BODY_POS \
               or obs_type is PyBulletObservationType.BODY_LIN_VEL \
               or obs_type is PyBulletObservationType.BODY_ANG_VEL:
                model_id = self._model_map[name]
                if obs_type is PyBulletObservationType.BODY_POS:
                    t, q = pybullet.getBasePositionAndOrientation(model_id)
                    data_obs += t + q
                else:
                    v, w = pybullet.getBaseVelocity(model_id)
                    if obs_type is PyBulletObservationType.BODY_LIN_VEL:
                        data_obs += v
                    else:
                        data_obs += w
            elif obs_type is PyBulletObservationType.LINK_POS \
                    or obs_type is PyBulletObservationType.LINK_LIN_VEL \
                    or obs_type is PyBulletObservationType.LINK_ANG_VEL:
                model_id, link_id = self._link_map[name]

                if obs_type is PyBulletObservationType.LINK_POS:
                    link_data = pybullet.getLinkState(model_id, link_id)
                    t = link_data[0]
                    q = link_data[1]
                    data_obs += t + q
                elif obs_type is PyBulletObservationType.LINK_LIN_VEL:
                    data_obs += pybullet.getLinkState(model_id, link_id, computeLinkVelocity=True)[-2]
                elif obs_type is PyBulletObservationType.LINK_ANG_VEL:
                    data_obs += pybullet.getLinkState(model_id, link_id, computeLinkVelocity=True)[-1]
            else:
                model_id, joint_id = self._joint_map[name]
                pos, vel, _, _ = pybullet.getJointState(model_id, joint_id)
                if obs_type is PyBulletObservationType.JOINT_POS:
                    data_obs.append(pos)
                elif obs_type is PyBulletObservationType.JOINT_VEL:
                    data_obs.append(vel)

        return np.array(data_obs)

    def _apply_control(self, action):

        i = 0
        for model_id, joint_id, mode in self._action_data:
            u = action[i]
            if mode is pybullet.POSITION_CONTROL:
                kwargs = dict(targetPosition=u, maxVelocity=pybullet.getJointInfo(model_id, joint_id)[11])
            elif mode is pybullet.VELOCITY_CONTROL:
                kwargs = dict(targetVelocity=u, maxVelocity=pybullet.getJointInfo(model_id, joint_id)[11])
            elif mode is pybullet.TORQUE_CONTROL:
                kwargs = dict(force=u)
            else:
                raise NotImplementedError

            pybullet.setJointMotorControl2(model_id, joint_id, mode, **kwargs)
            i += 1

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
            The action to be set in the actual pybullet simulation.

        """
        return action

    def _simulation_pre_step(self):
        """
        Allows information to be accesed and changed at every intermediate step
            before taking a step in the pybullet simulation.
            Can be usefull to apply an external force/torque to the specified bodies.
        """
        pass

    def _simulation_post_step(self):
        """
        Allows information to be accesed at every intermediate step
            after taking a step in the pybullet simulation.
            Can be usefull to average forces over all intermediate steps.
        """
        pass

    def _step_finalize(self):
        """
        Allows information to be accesed at the end of a step.
        """
        pass

    def _custom_load_models(self):
        """
        Allows to custom load a set of objects in the simulation

        Returns:
            A dictionary with the names and the ids of the loaded objects
        """
        return list()

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
