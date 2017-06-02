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

    @staticmethod
    def get_value(idx):
        return idx

    @staticmethod
    def get_idx(idx):
        return idx


class Discrete(Space):
    def __init__(self, n=None, values=None):
        if values is not None:
            self._values = np.unique(values).reshape(-1, 1)
            n = len(self._values)
        else:
            self._values = np.array([np.arange(n)]).reshape(-1, 1)

        self.n = n

    @property
    def dim(self):
        return 1

    @property
    def shape(self):
        return [len(self.values)]

    @property
    def values(self):
        return self._values

    def get_value(self, idx):
        return self._values[idx]

    def get_idx(self, value):
        return np.array([np.argwhere(self._values == value).ravel()[0]])


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
            shape.append(d.shape[0])

        return shape

    def get_value(self, idx):
        value = list()
        for s, i in enumerate(idx):
            value += self.discrete_spaces[s][i]

        return value

    def get_idx(self, value):
        idx = list()
        for s, v in enumerate(value.ravel()):
            idx.append(np.argwhere(
                self.discrete_spaces[s].values == [[v]]).ravel()[0])

        return idx
