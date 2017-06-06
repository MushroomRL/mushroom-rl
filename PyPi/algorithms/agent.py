import logging


class Agent(object):
    """
    This class implements the functions to evaluate the Q-function
    of the agent and drawing actions.
    """
    def __init__(self, approximator, policy, **params):
        """
        Constructor.

        # Arguments
            approximator (object): the approximator of the Q function.
            policy (object): the policy to use.
        """
        self.approximator = approximator
        self.policy = policy
        self.mdp_info = dict()
        self.logger = logging.getLogger('logger')
        self.params = params

        self.maxQs = list()

    def initialize(self, mdp_info):
        for k, v in mdp_info.iteritems():
            self.mdp_info[k] = v

    def draw_action(self, state, approximator):
        return self.policy(state, approximator)
