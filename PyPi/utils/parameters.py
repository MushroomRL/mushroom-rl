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
        if isinstance(idx, list):
            assert len(idx) == 2

            idx = np.concatenate((idx[0].astype(np.int),
                                  idx[1].astype(np.int)),
                                 axis=1).ravel()
        else:
            idx = idx.astype(np.int)
        assert idx.ndim == 1

        idx = tuple(idx) if idx.size == self._n_updates.ndim else 0

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
