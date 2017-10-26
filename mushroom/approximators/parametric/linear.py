import numpy as np


class LinearApproximator:
    def __init__(self, params=None, params_shape=None):
        if params is not None:
            self._w = params
        elif params_shape is not None:
            self._w = np.zeros(params_shape)
        else:
            raise ValueError('You should specify the initial parameter vector or the parameter shape')

    def fit(self, x, y, **fit_params):
        self._w = np.solve(x,y)

    def predict(self, x, **predict_params):
        return np.dot(x, self._w)

    def get_weights(self):
        return self._w

    def set_weights(self, w):
        self._w = w

    @property
    def weights_shape(self):
        return self._w.shape

    def diff(self, x):
        if len(self._w.shape) == 1 or self._w.shape[1] == 1:
            return x
        else:
            return np.array([x]*self._w.shape[1])
