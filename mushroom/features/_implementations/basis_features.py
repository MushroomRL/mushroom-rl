import numpy as np

from features_implementation import FeaturesImplementation


class BasisFeatures(FeaturesImplementation):
    def __init__(self, basis):
        self._basis = basis

    def __call__(self, *x):
        if len(x) > 1:
            x = np.concatenate(x, axis=0)
        else:
            x = x[0]

        out = np.empty(self.size)

        for i, bf in enumerate(self._basis):
            out[i] = bf(x)

        return out

    @property
    def size(self):
        return len(self._basis)
