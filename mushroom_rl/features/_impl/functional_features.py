from mushroom_rl.features.features import Features


class FunctionalFeatures(Features):
    def __init__(self, feature_list=None, n_outputs=None, function=None, backend='numpy'):
        self._n_outputs = n_outputs
        self._function = function

        super().__init__(backend, internal_backend=backend)

        self._add_save_attr(_n_outputs='primitive', _function='pickle')

    @property
    def size(self):
        return self._n_outputs

    def _compute(self, x):
        if self._function is not None:
            return self._function(x)

        return x
