import numpy as np


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
        self.values = np.arange(n)
        self.n = n

    @property
    def size(self):
        return self.n,

    @property
    def shape(self):
        return 1,
