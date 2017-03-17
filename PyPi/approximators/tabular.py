import numpy as np


class Tabular(object):
    """
    Tabular regressor. Used for discrete state and action spaces.
    """
    def __init__(self, **approximator_params):
        """
        Constructor.

        # Arguments
            approximator_params (dict): parameters.
        """
        self.approxim
        self._Q = np.zeros(approximator_params['shape'])

    def fit(self, x, y, _):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).
            y (np.array): target.
        """
        assert x.shape[1] == len(self._Q.shape), 'tabular regressor dimension ' \
                                                 'does not fit with input size.'

        dim = len(self._Q.shape)
        if dim > 1:
            idxs = list()
            for i in range(dim):
                idxs.append(x[:, i].astype(np.int))

            self._Q[idxs] = y
        else:
            self._Q[x] = y

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        assert x.shape[1] == len(self._Q.shape), 'tabular regressor dimension ' \
                                                 'does not fit with input size.'

        dim = len(self._Q.shape)
        if dim > 1:
            idxs = list()
            for i in range(dim):
                idxs.append(x[:, i].astype(np.int))

            return self._Q[idxs]
        else:
            return self._Q[x]
