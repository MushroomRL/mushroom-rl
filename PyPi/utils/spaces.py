import numpy as np

from gym import Space
from sklearn.utils.extmath import cartesian


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

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)


class Discrete(Space):
    def __init__(self, n):
        self.n_list = np.array([n]) if isinstance(n, int) else np.array(n)
        self.values = cartesian([np.arange(x, dtype=int) for x in self.n_list])
        self.n = np.prod(self.n_list)

    def sample(self):
        return np.array([np.random.choice(x) for x in self.n_list])

    @property
    def size(self):
        return tuple([x for x in self.n_list])

    @property
    def dim(self):
        return len(self.n_list)
