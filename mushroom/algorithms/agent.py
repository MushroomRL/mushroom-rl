class Agent(object):
    """
    This class implements the functions to evaluate the Q-function
    of the agent and drawing actions.

    """
    def __init__(self, approximator, policy, gamma, **params):
        """
        Constructor.

        Args:
            approximator (object): the approximator of the Q function;
            policy (object): the policy to use;
            gamma (float): discount factor;
            **params (dict): other parameters of the algorithm.

        """
        self.approximator = approximator
        self.policy = policy
        self._gamma = gamma
        self.params = params

        self.mdp_info = dict()

        self._next_action = None

    def initialize(self, mdp_info):
        """
        Fill the dictionary with information about the MDP.

        Args:
            mdp_info (dict): MDP information.

        """
        for k, v in mdp_info.iteritems():
            self.mdp_info[k] = v

    def draw_action(self, state, approximator=None):
        """
        Return the action to execute. It is the action returned by the policy
        or the action set by the algorithm (e.g. SARSA).

        Args:
            state (np.array): the state where the agent is;
            approximator (object): the approximator to use to draw the action.

        Returns:
            The action to be executed.

        """
        if self._next_action is None:
            if approximator is None:
                return self.policy(state, self.approximator)
            else:
                return self.policy(state, approximator)
        else:
            action = self._next_action
            self._next_action = None
            return action

    def episode_start(self):
        """
        Reset some parameters when a new episode starts. It is used only by
        some algorithms (e.g. DQN).

        """
        pass
