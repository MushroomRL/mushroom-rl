import numpy as np


class LinearApproximator:
    def __init__(self, params=None, input_dim=None, output_dim=1):
        if params is not None:
            self._w = params.reshape((output_dim, -1))
        elif input_dim is not None:
            self._w = np.zeros((output_dim, input_dim))
        else:
            raise ValueError('You should specify the initial parameter vector'
                             'or the input dimension')

    def fit(self, x, y, **fit_params):
        self._w = np.solve(x, y).T

    def predict(self, x, **predict_params):
        return np.dot(x, self._w.T)

    def get_weights(self):
        return self._w.flatten()

    def set_weights(self, w):
        self._w = w.reshape(self._w.shape)

    @property
    def weights_size(self):
        return self._w.size

    def diff(self, x):
        if len(self._w.shape) == 1 or self._w.shape[0] == 1:
            return x
        else:
            n_phi = self._w.shape[1]
            n_outs = self._w.shape[0]

            shape = (n_phi*n_outs, n_outs)
            df = np.zeros(shape)

            start = 0
            for i in xrange(n_outs):
                end = start + n_phi
                df[start:end, i] = x
                start = end
            return df
