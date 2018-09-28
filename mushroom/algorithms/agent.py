class Agent(object):
    """
    This class implements the functions to manage the agent (e.g. move the agent
    following its policy).

    """
    def __init__(self, policy, mdp_info, features=None):
        """
        Constructor.

        Args:
            policy (Policy): the policy followed by the agent;
            mdp_info (MDPInfo): information about the MDP;
            features (object, None): features to extract from the state.

        """
        self.policy = policy
        self.mdp_info = mdp_info

        self.phi = features

        self.next_action = None

    def fit(self, dataset):
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
