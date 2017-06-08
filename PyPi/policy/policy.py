import numpy as np

from PyPi.utils.dataset import max_QA
from PyPi.utils.parameters import Parameter


class EpsGreedy(object):
    """
    Epsilon greedy policy.
    """
    def __init__(self, epsilon, observation_space, action_space):
        """
        Constructor.

        # Arguments
        epsilon (Parameter): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        discrete_actions (np.array): the values of the discrete actions.
        """
        self.__name__ = 'EpsGreedy'

        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon
        self.observation_space = observation_space
        self.action_space = action_space

    def __call__(self, state, approximator):
        """
        Compute an action according to the policy.

        # Arguments
            state (np.array): the state where the agent is.
            approximator (object): the approximator to use to compute the
                action values.

        # Returns
            The selected action.
        """
        if not np.random.uniform() < self._epsilon(state):
            _, max_action = max_QA(state, False, approximator,
                                   self.action_space.values)
            return max_action.ravel()

        return np.array([np.random.choice(range(self.action_space.n))])

    def set_epsilon(self, epsilon):
        """
        Setter.

        # Arguments
        epsilon (Parameter): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def __str__(self):
        return self.__name__
