from mushroom_rl.core.serialization import Serializable


class Agent(Serializable):
    """
    This class implements the functions to manage the agent (e.g. move the agent
    following its policy).

    """

    def __init__(self, mdp_info, policy, features=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            policy (Policy): the policy followed by the agent;
            features (object, None): features to extract from the state.

        """
        self.mdp_info = mdp_info
        self.policy = policy

        self.phi = features

        self.next_action = None

        self._preprocessors = list()
        self._logger = None

        self._add_save_attr(
            mdp_info='pickle',
            policy='mushroom',
            phi='pickle',
            next_action='numpy',
            _preprocessors='mushroom',
            _logger='none'
        )

    def fit(self, dataset, **info):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('Agent is an abstract class')

    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm (e.g. in the
        case of SARSA).

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.phi is not None:
            state = self.phi(state)

        if self.next_action is None:
            return self.policy.draw_action(state)
        else:
            action = self.next_action
            self.next_action = None

            return action

    def episode_start(self):
        """
        Called by the agent when a new episode starts.

        """
        self.policy.reset()

    def stop(self):
        """
        Method used to stop an agent. Useful when dealing with real world
        environments, simulators, or to cleanup environments internals after
        a core learn/evaluate to enforce consistency.

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
        Add preprocessor to the preprocessor list.
        The preprocessors are applied in order.

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
