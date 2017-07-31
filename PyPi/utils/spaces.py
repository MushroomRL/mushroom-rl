import numpy as np

from gym import Space
from sklearn.utils.extmath import cartesian


class Box(Space):
    def __init__(self, low, high, shape=None):
        if shape is None:
            assert low.shape == high.shape
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self._low = low + np.zeros(shape)
            self._high = high + np.zeros(shape)
            self._shape = shape

    def sample(self):
        return np.random.uniform(low=self._low,
                                 high=self._high,
                                 size=self._shape)

    @property
    def low(self):
        return np.unique(self._low)[0]

    @property
    def high(self):
        return np.unique(self._high)[0]


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
