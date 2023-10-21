from mushroom_rl.core.serialization import Serializable

from ._impl import *


class AgentInfo(Serializable):
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


class Agent(Serializable):
    """
    This class implements the functions to manage the agent (e.g. move the agent following its policy).

    """

    def __init__(self, mdp_info, policy, is_episodic=False, backend='numpy'):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            policy (Policy): the policy followed by the agent;
            is_episodic (bool, False): whether the agent is learning in an episodic fashion or not;
            backend (str, 'numpy'): array backend to be used by the algorithm.

        """
        self.mdp_info = mdp_info
        self._info = AgentInfo(
            is_episodic=is_episodic,
            policy_state_shape=policy.policy_state_shape,
            backend=backend
        )

        self.policy = policy
        self.next_action = None
        self._agent_converter = DataConversion.get_converter(backend)
        self._env_converter = DataConversion.get_converter(self.mdp_info.backend)

        self._preprocessors = list()

        self._logger = None

        self._add_save_attr(
            policy='mushroom',
            next_action='none',
            mdp_info='mushroom',
            _info='mushroom',
            _agent_converter='primitive',
            _env_converter='primitive',
            _preprocessors='mushroom',
            _logger='none'
        )

    def fit(self, dataset, **info):
        """
        Fit step.

        Args:
            dataset (Dataset): the dataset.

        """
        raise NotImplementedError('Agent is an abstract class')

    def draw_action(self, state, policy_state=None):
        """
        Return the action to execute in the given state. It is the action returned by the policy or the action set by
        the algorithm (e.g. in the case of SARSA).

        Args:
            state: the state where the agent is;
            policy_state: the policy internal state.

        Returns:
            The action to be executed.

        """
        if self.next_action is None:
            action, next_policy_state = self.policy.draw_action(state, policy_state)
        else:
            action = self.next_action
            next_policy_state = None  # FIXME
            self.next_action = None

        return self._convert_to_env_backend(action), self._convert_to_env_backend(next_policy_state)

    def episode_start(self, episode_info):
        """
        Called by the agent when a new episode starts.

         Args:
            episode_info (dict): a dictionary containing the information at reset, such as context.

        """
        return self.policy.reset()

    def stop(self):
        """
        Method used to stop an agent. Useful when dealing with real world environments, simulators, or to cleanup
        environments internals after a core learn/evaluate to enforce consistency.

        """
        pass

    def set_logger(self, logger):
        """
        Setter that can be used to pass a logger to the algorithm

        Args:
            logger (Logger): the logger to be used by the algorithm.

        """
        self._logger = logger

    def add_preprocessor(self, preprocessor):
        """
        Add preprocessor to the preprocessor list. The preprocessors are applied in order.

        Args:
            preprocessor (object): state preprocessors to be applied
                to state variables before feeding them to the agent.

        """
        self._preprocessors.append(preprocessor)

    @property
    def preprocessors(self):
        """
        Access to state preprocessors stored in the agent.

        """
        return self._preprocessors

    def _convert_to_env_backend(self, array):
        return self._env_converter.to_backend_array(self._agent_converter, array)

    def _convert_to_agent_backend(self, array):
        return self._agent_converter.to_backend_array(self._env_converter, array)

    @property
    def info(self):
        return self._info

