import numpy as np


class EpsGreedy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        if np.random.uniform() > self.epsilon:
            return False
        return True