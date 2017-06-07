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
            values = np.unique(values)
            n = len(values)
        else:
            values = np.arange(n)

        self._idxs = dict()
        self._values = dict()
        for i, v in enumerate(values):
            self._idxs[v] = i
            self._values[i] = v

        self.n = n

    @property
    def dim(self):
        return 1

    @property
    def shape(self):
        return self.n

    @property
    def values(self):
        return np.array(self._values.values()).reshape(-1, 1)

    def get_value(self, idx):
        if idx.ndim == 1:
            assert idx.size == 1

            values = [self._values[idx[0]]]
        elif idx.ndim == 2:
            assert idx.shape[1] == 1

            values = [[self._values[i[0]]] for i in idx]
        else:
            raise ValueError('Wrong dimension for indices array.')

        return np.array(values)

    def get_idx(self, value):
        if value.ndim == 1:
            assert value.size == 1

            idxs = [self._idxs[value[0]]]
        elif value.ndim == 2:
            assert value.shape[1] == 1

            idxs = [[self._values[v[0]]] for v in value]
        else:
            raise ValueError('Wrong dimension for indices array.')

        return np.array(idxs)


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

    def get_value(self, idx):
        if idx.ndim == 1:
            values = [
                self.discrete_spaces[i]._values[j] for i, j in enumerate(idx)]
        elif idx.ndim == 2:
            values = list()
            for j in idx:
                value = [
                    self.discrete_spaces[i]._idxs[j1] for i, j1 in enumerate(j)]
                values.append(value)
        else:
            raise ValueError('Wrong dimension for indices array.')

        return np.array(values)

    def get_idx(self, value):
        if value.ndim == 1:
            idxs = [
                self.discrete_spaces[i]._idxs[v] for i, v in enumerate(value)]
        elif value.ndim == 2:
            idxs = list()
            for v in value:
                idx = [
                    self.discrete_spaces[i]._idxs[v1] for i, v1 in enumerate(v)]
                idxs.append(idx)
        else:
            raise ValueError('Wrong dimension for indices array.')

        return np.array(idxs)
