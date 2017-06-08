import numpy as np

from gym import Space


class Box(Space):
    def __init__(self, low, high, shape=None):
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
            self.shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
            self.shape = shape

    @property
    def dim(self):
        return self.shape[0]


class Discrete(Space):
    def __init__(self, n):
        self.n = n

    @property
    def dim(self):
        return 1

    @property
    def shape(self):
        return self.n

    @property
    def values(self):
        return np.arange(self.n).reshape(-1, 1)


class MultiDiscrete(Space):
    def __init__(self, discrete_spaces):
        self.discrete_spaces = discrete_spaces

    @property
    def dim(self):
        return len(self.discrete_spaces)

    @property
    def shape(self):
        shape = list()
        for d in self.discrete_spaces:
            shape.append(d.shape)

        return shape
