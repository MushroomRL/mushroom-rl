import numpy as np


class BasisFeatures:
    def __init__(self, basis):
        self._basis = basis

    def __call__(self, *x):
        out = np.empty(self.size)

        for i, bf in enumerate(self._basis):
            out[i] = bf(*x)

        return out

    @property
    def size(self):
        return len(self._basis)

