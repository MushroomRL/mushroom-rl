import numpy as np


class EpsGreedy(object):
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def __call__(self):
        if np.random.uniform() < self._epsilon:
            return False
        return True

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
