import numpy as np
from sklearn.utils.extmath import cartesian

import gym
from gym import spaces as gymspaces
from gym.spaces import prng


class Box(gymspaces.Box):
    def __init__(self, low, high, shape=None):
        super(Box, self).__init__(low, high, shape)

    @property
    def dim(self):
        return self.shape


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


class DiscreteValued(gym.Space):
    def __init__(self, values):
        if isinstance(values, list):
            self._values = np.array(values)
        elif isinstance(values, np.array):
            self._values = values
        else:
            ValueError('Unrecognized values vector.')

    def sample(self):
        return prng.np_random.choice(self._values)

    def contains(self, x):
        if x in self._values:
            return True
        else:
            return False

    def __repr__(self):
        return "DiscreteValued(%d)" % self.n_values

    def __eq__(self, other):
        return self._values == other.values

    @property
    def dim(self):
        return 1

    @property
    def values(self):
        return self._values

    @property
    def n_values(self):
        return self._values.size()


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
