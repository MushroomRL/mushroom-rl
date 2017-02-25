import numpy as np


class Tabular(object):
    def __init__(self, **apprx_params):
        self._Q = np.zeros(apprx_params['shape'])

    def fit(self, x, y):
        dim = len(self._Q.shape)
        if dim > 1:
            idxs = list()
            for i in range(dim):
                idxs.append(x[:, i].astype(np.int))

            self._Q[idxs] = y
        else:
            self._Q[x] = y

    def predict(self, x):
        dim = len(self._Q.shape)
        if dim > 1:
            idxs = list()
            for i in range(dim):
                idxs.append(x[:, i].astype(np.int))

            return self._Q[idxs]
        else:
            return self._Q[x]
