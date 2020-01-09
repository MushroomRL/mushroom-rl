from .features_implementation import FeaturesImplementation


class FunctionalFeatures(FeaturesImplementation):
    def __init__(self, n_outputs, function):
        self._n_outputs = n_outputs
        self._function = function if function is not None else lambda x: x

    def __call__(self, *args):
        x = self._concatenate(args)

        return self._function(x)

    @property
    def size(self):
        return self._n_outputs
