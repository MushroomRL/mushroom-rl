from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.history_manager import HistoryManager


class AgentInfo(MushroomObject):
    def __init__(self, is_episodic, policy_state_shape, backend):
        assert isinstance(is_episodic, bool)
        assert policy_state_shape is None or isinstance(policy_state_shape, tuple)
        assert isinstance(backend, str)

        self.is_episodic = is_episodic
        self.is_stateful = policy_state_shape is not None
        self.policy_state_shape = policy_state_shape
        self.backend = backend

        self._add_save_attr(
            is_episodic='primitive',
            is_stateful='primitive',
            policy_state_shape='primitive',
            backend='primitive'
        )


class Agent(MushroomObject):
    """
    This class implements the functions to manage the agent (e.g. move the agent following its policy).

    """

    def __init__(self, mdp_info, policy, is_episodic=False, backend='numpy', history_length=1):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            policy (Policy): the policy followed by the agent;
            is_episodic (bool, False): whether the agent is learning in an episodic fashion or not;
            backend (str, 'numpy'): array backend to be used by the algorithm;
            history_length (int, 1): number of states to stack as input to the policy. When > 1 the agent maintains a
                rolling observation buffer (not stored in the dataset).

        """
        self.mdp_info = mdp_info
        self._info = AgentInfo(
            is_episodic=is_episodic,
            policy_state_shape=policy.policy_state_shape if policy.is_stateful else None,
            backend=backend
        )
        self.policy = policy
        self.next_action = None
        self._agent_backend = ArrayBackend.get_array_backend(backend)
        self._env_backend = ArrayBackend.get_array_backend(self.mdp_info.backend)

        self._history_manager = self._build_history_manager(history_length)

        self._core_preprocessors = list()
        self._agent_preprocessors = list()

        self._add_logger_attr('policy')

        self._add_save_attr(
            policy='mushroom',
            next_action='none',
            mdp_info='mushroom',
            _info='mushroom',
            _history_manager='mushroom',
            _agent_backend='primitive',
            _env_backend='primitive',
            _core_preprocessors='mushroom',
            _agent_preprocessors='mushroom'
        )

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (Dataset): the dataset.

        """
        raise NotImplementedError('Agent is an abstract class')

    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action returned by the policy or the action set by
        the algorithm (e.g. in the case of SARSA). When the policy is stateful, its internal state is updated and can
        be read through :meth:`get_policy_state`.

        Args:
            state: the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.next_action is None:
            state = self._convert_to_agent_backend(state)
            state = self._agent_preprocess(state)

            if self._history_manager is not None:
                state = self._history_manager(state)

            action = self.policy.draw_action(state)
        else:
            action = self.next_action
            self.next_action = None

        return self._convert_to_env_backend(action)

    def get_policy_state(self):
        """
        Returns:
            The current internal state of the policy, converted to the environment backend, or ``None`` if the policy
            is stateless.

        """
        if self.policy.is_stateful:
            return self._convert_to_env_backend(self.policy.policy_state)
        return None

    def episode_start(self, initial_state, episode_info):
        """
        Called by the Core when a new episode starts.

        Args:
            initial_state (Array): vector representing the initial state of the environment.
            episode_info (dict): a dictionary containing the information at reset, such as context.

        Returns:
            A tuple containing the policy initial state and, optionally, the policy parameters

        """
        if self._history_manager is not None:
            self._history_manager.reset()

        return self._convert_to_env_backend(self.policy.reset()), None

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        """
        Called by the Core at the start of a new episode when using a vectorized environment.

        Args:
            initial_states (Array): the initial states of the environment.
            episode_info (dict): a dictionary containing the information at reset, such as context;
            start_mask (Array): boolean mask to select the environments that are starting a new episode

        Returns:
            A tuple containing the policy initial states and, optionally, the policy parameters

        """
        if self._history_manager is not None:
            self._history_manager.reset_vectorized(start_mask)

        return self._convert_to_env_backend(self.policy.reset_vectorized(start_mask)), None

    def stop(self):
        """
        Method used to stop an agent. Useful when dealing with real world environments, simulators, or to cleanup
        environments internals after a core learn/evaluate to enforce consistency.

        """
        if self.policy.is_stateful:
            self.policy.stop()

    def add_core_preprocessor(self, preprocessor):
        """
        Add preprocessor to the core's preprocessor list. The preprocessors are applied in order.

        Args:
            preprocessor (object): state preprocessors to be applied
                to state variables before feeding them to the agent.

        """
        self._core_preprocessors.append(preprocessor)

    def add_agent_preprocessor(self, preprocessor):
        """
        Add preprocessor to the agent's preprocessor list. The preprocessors are applied in order.

        Args:
            preprocessor (object): state preprocessors to be applied
                to state variables before feeding them to the agent.

        """
        self._agent_preprocessors.append(preprocessor)

    @property
    def core_preprocessors(self):
        """
        Access to core's state preprocessors stored in the agent.

        """
        return self._core_preprocessors

    def _build_history_manager(self, history_length):
        if history_length > 1:
            obs_shape = self.mdp_info.observation_space.shape
            obs_dtype = self.mdp_info.observation_space.data_type
            return HistoryManager(history_length, obs_shape, obs_dtype, self._env_backend, self._agent_backend)
        return None

    def _convert_to_env_backend(self, array):
        return self._env_backend.convert_to_backend(self._agent_backend, array)

    def _convert_to_agent_backend(self, array):
        return self._agent_backend.convert_to_backend(self._env_backend, array)

    def _agent_preprocess(self, state):
        """
        Applies all the agent's preprocessors to the state.

        Args:
            state (Array): the state where the agent is;

        Returns:
            The preprocessed state.

        """
        for p in self._agent_preprocessors:
            state = p(state)
        return state

    def _update_agent_preprocessor(self, state):
        """
        Updates the stats of all the agent's preprocessors given the state.

        Args:
            state (Array): the state where the agent is;

        """
        for i, p in enumerate(self._agent_preprocessors, 1):
            p.update(state)
            if i < len(self._agent_preprocessors):
                state = p(state)

    @property
    def info(self):
        return self._info

    @property
    def history_length(self):
        """
        The number of observations stacked as policy input, ``1`` when no history is used.

        """
        return self._history_manager.history_length if self._history_manager is not None else 1

