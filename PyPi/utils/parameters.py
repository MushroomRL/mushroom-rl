import numpy as np


class Parameter(object):
    def __init__(self, value, decay=False, decay_exp=1., min_value=None,
                 shape=(1,)):
        self.value = np.ones(shape) * value
        self._decay = decay
        self._decay_exp = decay_exp
        self._min_value = min_value
        self._n_updates = np.ones(shape)

    def __call__(self, idx):
        if self.value.shape != (1,):
            idx_list = idx[0].astype('int').ravel().tolist()
            for i in idx[1:]:
                idx_list += i.astype('int').ravel().tolist()
            idx = tuple(idx_list)
        else:
            idx = (np.array([0]))

        value = self.value[idx]
        self._update(idx)

        return value

    def _update(self, idx):
        if self._decay:
            new_value = 1. / self._n_updates[idx] ** self._decay_exp
            if new_value >= self._min_value:
                self.value[idx] = new_value

        self._n_updates[idx] += 1
