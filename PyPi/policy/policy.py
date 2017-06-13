import numpy as np

from PyPi.utils.dataset import max_QA
from PyPi.utils.parameters import Parameter


class EpsGreedy(object):
    """
    Epsilon greedy policy.
    """
    def __init__(self, epsilon, discrete_actions):
        """
        Constructor.

        # Arguments
        epsilon (0 <= float <= 1): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        self.__name__ = 'EpsGreedy'

        self._epsilon = Parameter(epsilon)
        self.discrete_actions = discrete_actions

    def __call__(self, states, approximator):
        """
        Compute an action according to the policy.

        # Arguments
            states (np.array): the state where the agent is.
            absorbing (np.array): whether the state is absorbing or not.
            force_max_action (bool): whether to select the best action or not.

        # Returns
            The selected action.
        """
        if not np.random.uniform() < self._epsilon():
            _, max_action = max_QA(states, False, approximator,
                                   self.discrete_actions)
            return max_action

        return np.array([np.random.choice(self.action_space.n)])

    def set_epsilon(self, epsilon):
        """
        Setter.

        # Arguments
        epsilon (0 <= float <= 1): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def update(self):
        self._epsilon.update()

    def __str__(self):
        return self.__name__
