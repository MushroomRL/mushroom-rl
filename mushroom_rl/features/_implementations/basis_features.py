import numpy as np

from .features_implementation import FeaturesImplementation


class BasisFeatures(FeaturesImplementation):
    def __init__(self, basis):
        self._basis = basis

    def __call__(self, *args):
        x = self._concatenate(args)

        y = list()

        x = np.atleast_2d(x)
        for s in x:
            out = np.empty(self.size)

            for i, bf in enumerate(self._basis):
                out[i] = bf(s)

            y.append(out)

        if len(y) == 1:
            y = y[0]
        else:
            y = np.array(y)

        return y

    @property
    def size(self):
        return len(self._basis)
