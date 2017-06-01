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
        epsilon (Parameter): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        discrete_actions (np.array): the values of the discrete actions.
        """
        self.__name__ = 'EpsGreedy'

        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon
        self.discrete_actions = discrete_actions

    def __call__(self, states, approximator):
        """
        Compute an action according to the policy.

        # Arguments
            states (np.array): the state where the agent is.
            approximator (object): the approximator to use to compute the
                action values.

        # Returns
            The selected action.
        """
        if not np.random.uniform() < self._epsilon():
            _, max_action = max_QA(states, False, approximator,
                                   self.discrete_actions)
            return max_action
        return np.array([self.discrete_actions[
            np.random.choice(range(self.discrete_actions.shape[0])), :]])

    def set_epsilon(self, epsilon):
        """
        Setter.

        # Arguments
        epsilon (Parameter): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def update(self):
        """
        Update epsilon.
        """
        self._epsilon.update()

    def __str__(self):
        return self.__name__
