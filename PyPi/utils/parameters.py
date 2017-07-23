import numpy as np


class Parameter(object):
    def __init__(self, value, min_value=None, shape=(1,)):
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = np.zeros(shape)

    def __call__(self, idx, **kwargs):
        if isinstance(idx, list):
            assert idx[0].ndim == 2 and idx[1].ndim == 2
            assert idx[0].shape[0] == idx[1].shape[0]

            idx = tuple(np.concatenate((idx[0], idx[1]), axis=1).ravel())
        else:
            idx = tuple(idx)

        idx = idx if self._n_updates.size > 1 else 0

        self._n_updates[idx] += 1

        self._update(idx, **kwargs)

        return self.get_value(idx, **kwargs)

    def get_value(self, idx, **kwargs):
        new_value = self._compute(idx, **kwargs)

        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, idx, **kwargs):
        return self._initial_value

    def _update(self, idx, **kwargs):
        pass


class DecayParameter(Parameter):
    def __init__(self, value, decay_exp=1., min_value=None,
                 shape=(1,)):
        self._decay_exp = decay_exp

        super(DecayParameter, self).__init__(value, min_value, shape)

    def _compute(self, idx, **kwargs):
        return self._initial_value / self._n_updates[idx] ** self._decay_exp

    def _update(self, idx, **kwargs):
        pass
