import numpy as np


class Parameter(object):
    def __init__(self, value, decay=False, decay_exp=1., min_value=None,
                 shape=(1,)):
        self._initial_value = value
        self._decay = decay
        self._decay_exp = decay_exp
        self._min_value = min_value
        self._n_updates = np.zeros(shape)

    def __call__(self, idx, **kwargs):
        if not isinstance(idx, tuple):
            idx = tuple(idx.ravel())

        idx = idx if self._n_updates.size > 1 else 0

        self._n_updates[idx] += 1

        return self._compute(idx)

    def _compute(self, idx):
        if self._decay:
            new_value =\
                self._initial_value / self._n_updates[idx] ** self._decay_exp
            if self._min_value is None or new_value >= self._min_value:
                return new_value
            else:
                return self._min_value
        else:
            return self._initial_value
