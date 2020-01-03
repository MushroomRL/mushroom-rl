import numpy as np

from .features_implementation import FeaturesImplementation


class FunctionalFeatures(FeaturesImplementation):
    def __init__(self, n_outputs, function):
        self._n_outputs = n_outputs
        self._function = function if function is not None else lambda x: x

    def __call__(self, *args):
        x = self._concatenate(args)

        return self._function(x)

    @staticmethod
    def _concatenate(args):
        if len(args) > 1:
            x = np.concatenate(args, axis=-1)
        else:
            x = args[0]

        return x

    @property
    def size(self):
        return self._n_outputs
