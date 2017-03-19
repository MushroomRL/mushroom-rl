import numpy as np


class EpsGreedy(object):
    """
    Epsilon greedy policy.
    """
    def __init__(self, epsilon):
        """
        Constructor.

        # Arguments
        epsilon (0 <= float <= 1): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        self.__name__ = 'EpsGreedy'

        self._epsilon = epsilon

    def __call__(self):
        """
        # Returns
            Flag indicating to perform the greedy action or the random one.
        """
        if np.random.uniform() < self._epsilon:
            return False
        return True

    def set_epsilon(self, epsilon):
        """
        Setter.

        # Arguments
        epsilon (0 <= float <= 1): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step.
        """
        self._epsilon = epsilon

    def __str__(self):
        return self.__name__
