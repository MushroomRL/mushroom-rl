import numpy as np


class Parameter(object):
    def __init__(self, value, min_value=None, shape=(1,)):
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = np.zeros(shape)

    def __call__(self, idx, **kwargs):
        if self._n_updates.size == 1:
            idx = 0
        else:
            if isinstance(idx, list):
                assert idx[0].ndim == 2 and idx[1].ndim == 2
                assert idx[0].shape[0] == idx[1].shape[0]

                idx = tuple(np.concatenate((idx[0], idx[1]), axis=1).ravel())
            else:
                idx = tuple(idx.ravel())

            idx = tuple([int(i) for i in idx])

        self.update(idx)

        return self.get_value(idx, **kwargs)

    def get_value(self, idx=0, **kwargs):
        new_value = self._compute(idx, **kwargs)

        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, idx, **kwargs):
        return self._initial_value

    def update(self, idx=0):
        self._n_updates[idx] += 1

    @property
    def shape(self):
        return self._n_updates.shape


class LinearDecayParameter(Parameter):
    def __init__(self, value,  min_value, n, shape=(1,)):
        self._coeff = (min_value - value) / n

        super(LinearDecayParameter, self).__init__(value, min_value, shape)

    def _compute(self, idx, **kwargs):
        return self._coeff * self._n_updates[idx] + self._initial_value


class DecayParameter(Parameter):
    def __init__(self, value, decay_exp=1., min_value=None, shape=(1,)):
        self._decay_exp = decay_exp

        super(DecayParameter, self).__init__(value, min_value, shape)

    def _compute(self, idx, **kwargs):
        return self._initial_value / self._n_updates[idx] ** self._decay_exp
