import numpy as np


class Parameter(object):
    def __init__(self, value, decay=False, decay_exp=1., min_value=None,
                 shape=(1,)):
        self._initial_value = value
        self._decay = decay
        self._decay_exp = decay_exp
        self._min_value = min_value
        self._n_updates = np.zeros(shape)

        self.value = value

    def __call__(self, idx):
        assert idx.ndim == 1 or idx.shape[0] == 1

        idx = tuple(idx.ravel()) if idx.shape != (1,) else 0

        self._update(idx)

        return self.value

    def _update(self, idx):
        self._n_updates[idx] += 1
        if self._decay:
            new_value =\
                self._initial_value / self._n_updates[idx] ** self._decay_exp
            if self._min_value is None or new_value >= self._min_value:
                self.value = new_value
