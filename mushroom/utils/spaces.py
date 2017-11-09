import numpy as np

from sklearn.utils.extmath import cartesian


class Box:
    def __init__(self, low, high, shape=None):
        if shape is None:
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = low
            self._high = high
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

    def sample(self):
        return np.random.uniform(low=self._low,
                                 high=self._high,
                                 size=self._shape)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def shape(self):
        return self._shape


class Discrete:
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
    def shape(self):
        return (len(self.n_list),)
