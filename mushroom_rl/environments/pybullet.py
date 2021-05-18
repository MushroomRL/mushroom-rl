import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.pybullet import *


class PyBullet(Environment):
    """
    Class to create a Mushroom environment using the PyBullet simulator.

    """
    def __init__(self, files, actuation_spec, observation_spec, gamma,
                 horizon, timestep=1/240, n_intermediate_steps=1, enforce_joint_velocity_limits=False,
                 debug_gui=False, **viewer_params):
        """
        Constructor.

        Args:
            files (dict): dictionary of the URDF/MJCF/SDF files to load (key) and parameters dictionary (value);
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
            enforce_joint_velocity_limits (bool, False): flag to enforce the velocity limits;
            debug_gui (bool, False): flag to activate the default pybullet visualizer, that can be used for debug
                purposes;
            **viewer_params: other parameters to be passed to the viewer.
                See PyBulletViewer documentation for the available options.

        """
        assert len(observation_spec) > 0
        assert len(actuation_spec) > 0

        # Store simulation parameters
        self._timestep = timestep
        self._n_intermediate_steps = n_intermediate_steps

        # Create the simulation and viewer
        if debug_gui:
            self._client = BulletClient(connection_mode=pybullet.GUI)
            self._client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self._client = BulletClient(connection_mode=pybullet.DIRECT)
        self._client.setTimeStep(self._timestep)
        self._client.setGravity(0, 0, -9.81)
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._viewer = PyBulletViewer(self._client, self.dt, **viewer_params)
        self._state = None

        # Load model and create access maps
        self._model_map = dict()
        for file_name, kwargs in files.items():
            model_id = self._load_model(file_name, kwargs)
            model_name = self._client.getBodyInfo(model_id)[1].decode('UTF-8')
            self._model_map[model_name] = model_id
        self._model_map.update(self._custom_load_models())

        observation_space = Box(*self._indexer.observation_limits)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Let the child class modify the mdp_info data structure
        mdp_info = self._modify_mdp_info(mdp_info)

        # Provide the structure to the superclass
        super().__init__(mdp_info)

        # Save initial state of the MDP
        self._initial_state = self._client.saveState()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        self._client.restoreState(self._initial_state)
        self.setup(state)
        self._state = self._create_sim_state()
        return self._state

    def render(self):
        self._viewer.display()

    def stop(self):
        pass

    def step(self, action):
        curr_state = self._state.copy()

        action = self._preprocess_action(action)

        self._step_init(curr_state, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(curr_state, action)
            self._indexer.apply_control(ctrl_action)

            self._simulation_pre_step()

            self._client.stepSimulation()

            self._simulation_post_step()

        self._state = self._create_sim_state()

        self._step_finalize()

        absorbing = self.is_absorbing(self._state)
        reward = self.reward(cur_obs, action, self._state, absorbing)

        observation = self._create_observation(self._state)

        return observation, reward, absorbing, {}

    def get_sim_state_index(self, name, obs_type):
        return self._observation_indices_map[name][obs_type]

    def get_sim_state(self, obs, name, obs_type):
        """
        Returns a specific observation value

        Args:
            obs (np.ndarray): the observation vector;
            name (str): the name of the object to consider;
            obs_type (PyBulletObservationType): the type of observation to be used.

        Returns:
            The required elements of the input state vector.

        """
        indices = self.get_sim_state_index(name, obs_type)

        return obs[indices]

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

    def _create_observation(self, state):
        """
        This method can be overridden to ctreate an observation vector from the simulator state vector.
        By default, returns the simulator state vector unchanged.

        Args:
            state (np.ndarray): the simulator state vector.

        Returns:
            The environment observation.

        """
        return state

    def _load_model(self, file_name, kwargs):
        if file_name.endswith('.urdf'):
            model_id = self._client.loadURDF(file_name, **kwargs)
        elif file_name.endswith('.sdf'):
            model_id = self._client.loadSDF(file_name, **kwargs)[0]
        else:
            model_id = self._client.loadMJCF(file_name, **kwargs)[0]

        return model_id

    def _compute_action_limits(self):
        low = list()
        high = list()

        for model_id, joint_id, mode in self._action_data:
            joint_info = self._client.getJointInfo(model_id, joint_id)
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
            index_count = len(low)
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
                joint_info = self._client.getJointInfo(model_id, joint_id)

                if obs_type is PyBulletObservationType.JOINT_POS:
                    low.append(joint_info[8])
                    high.append(joint_info[9])
                else:
                    max_joint_vel = joint_info[11]
                    low.append(-max_joint_vel)
                    high.append(max_joint_vel)

            self._add_observation_index(name, obs_type, index_count, len(low))

        return np.array(low), np.array(high)

    def _add_observation_index(self, name, obs_type, start, end):
        if name not in self._observation_indices_map:
            self._observation_indices_map[name] = dict()

        self._observation_indices_map[name][obs_type] = list(range(start, end))

    def _create_sim_state(self):
        data_obs = list()

        for name, obs_type in self._observation_map:
            if obs_type is PyBulletObservationType.BODY_POS \
               or obs_type is PyBulletObservationType.BODY_LIN_VEL \
               or obs_type is PyBulletObservationType.BODY_ANG_VEL:
                model_id = self._model_map[name]
                if obs_type is PyBulletObservationType.BODY_POS:
                    t, q = self._client.getBasePositionAndOrientation(model_id)
                    data_obs += t + q
                else:
                    v, w = self._client.getBaseVelocity(model_id)
                    if obs_type is PyBulletObservationType.BODY_LIN_VEL:
                        data_obs += v
                    else:
                        data_obs += w
            elif obs_type is PyBulletObservationType.LINK_POS \
                    or obs_type is PyBulletObservationType.LINK_LIN_VEL \
                    or obs_type is PyBulletObservationType.LINK_ANG_VEL:
                model_id, link_id = self._link_map[name]

                if obs_type is PyBulletObservationType.LINK_POS:
                    link_data = self._client.getLinkState(model_id, link_id)
                    t = link_data[0]
                    q = link_data[1]
                    data_obs += t + q
                elif obs_type is PyBulletObservationType.LINK_LIN_VEL:
                    data_obs += self._client.getLinkState(model_id, link_id, computeLinkVelocity=True)[-2]
                elif obs_type is PyBulletObservationType.LINK_ANG_VEL:
                    data_obs += self._client.getLinkState(model_id, link_id, computeLinkVelocity=True)[-1]
            else:
                model_id, joint_id = self._joint_map[name]
                pos, vel, _, _ = self._client.getJointState(model_id, joint_id)
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
                kwargs = dict(targetPosition=u, maxVelocity=self._client.getJointInfo(model_id, joint_id)[11])
            elif mode is pybullet.VELOCITY_CONTROL:
                kwargs = dict(targetVelocity=u, maxVelocity=self._client.getJointInfo(model_id, joint_id)[11])
            elif mode is pybullet.TORQUE_CONTROL:
                kwargs = dict(force=u)
            else:
                raise NotImplementedError

            self._client.setJointMotorControl2(model_id, joint_id, mode, **kwargs)
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

    def reward(self, state, action, next_state, absorbing):
        """
        Compute the reward based on the given transition.

        Args:
            state (np.array): the current state of the system;
            action (np.array): the action that is applied in the current state;
            next_state (np.array): the state reached after applying the given action;
            absorbing (bool): whether next_state is an absorbing state or not.

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

    def setup(self, state):
        """
        A function that allows to execute setup code after an environment
        reset.

        Args:
            state (np.ndarray): the state to be restored. If the state should be
            chosen by the environment, state is None. Environments can ignore this
            value if the initial state cannot be set programmatically.

        """
        pass

    @property
    def client(self):
        return self._client

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps

