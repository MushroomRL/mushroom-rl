import numpy as np

from mushroom_rl.features.features import Features


class BasisFeatures(Features):
    def __init__(self, feature_list=None, *, backend='numpy', **kwargs):
        self._basis = self._as_list(feature_list)

        super().__init__(feature_list=feature_list, backend=backend, **kwargs)

        self._add_save_attr(_basis='mushroom')

    @property
    def size(self):
        return len(self._basis)

    def _compute(self, x):
        x = np.atleast_2d(x)

        out = np.empty((x.shape[0], self.size))
        for i, bf in enumerate(self._basis):
            out[:, i] = bf(x)

        if out.shape[0] == 1:
            return out[0]

        return out

    @property
    def _internal_backend(self):
        return 'numpy'
