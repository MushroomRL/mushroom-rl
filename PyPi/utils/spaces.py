import numpy as np
from sklearn.utils.extmath import cartesian

import gym
from gym import spaces as gymspaces


class Discrete(gymspaces.Discrete):
    def __init__(self, n):
        super(Discrete, self).__init__(n)

    @property
    def dim(self):
        return 1

    @property
    def values(self):
        return np.array([np.arange(self.n)]).reshape(-1, 1)

    @property
    def n_values(self):
        return self.n


class MultiDiscrete(gymspaces.MultiDiscrete):
    def __init__(self, discrete_spaces):
        super(MultiDiscrete, self).__init__(discrete_spaces)

        values = list()
        self._n_values = list()
        self.n = 1
        for value in discrete_spaces:
            values.append(np.arange(value[0], value[1] + 1).tolist())
            n_elements = len(values[-1])
            self.n *= n_elements
            self._n_values.append(n_elements)
        self._values = np.asarray(cartesian(values))

    @property
    def dim(self):
        return self.shape

    @property
    def values(self):
        return self._values

    @property
    def n_values(self):
        return self._n_values


class DiscreteValued(gym.Space):
    pass
