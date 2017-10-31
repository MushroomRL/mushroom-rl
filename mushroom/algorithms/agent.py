class Agent(object):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, policy, gamma, params, features=None):
        """
        Constructor.

        Args:
            policy (object): the policy to use for the agent;
            gamma (float): discount factor;
            params (dict): other parameters of the algorithm;
            features (object, None): features to use for the input of the
                approximator.

        """
        self.policy = policy
        self._gamma = gamma
        self.params = params

        self.mdp_info = dict()

        self.phi = features

        self._next_action = None

    def initialize(self, mdp_info):
        """
        Fill the dictionary with information about the MDP.

        Args:
            mdp_info (dict): MDP information.

        """
        for k, v in mdp_info.iteritems():
            self.mdp_info[k] = v

    def draw_action(self, state):
        """
        Return the action to execute. It is the action returned by the policy
        or the action set by the algorithm (e.g. SARSA).

        Args:
            state (np.array): the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.phi is not None:
            state = self.phi(state)

        if self._next_action is None:
            return self.policy(state)
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
