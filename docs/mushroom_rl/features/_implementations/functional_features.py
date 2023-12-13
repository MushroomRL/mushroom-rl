from .features_implementation import FeaturesImplementation


class FunctionalFeatures(FeaturesImplementation):
    def __init__(self, n_outputs, function):
        self._n_outputs = n_outputs
        self._function = function if function is not None else self._identity

    def __call__(self, *args):
        x = self._concatenate(args)

        return self._function(x)

    def _identity(self, x):
        return x

    @property
    def size(self):
        return self._n_outputs
