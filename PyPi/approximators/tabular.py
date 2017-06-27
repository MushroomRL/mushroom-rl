import numpy as np


class Tabular(object):
    """
    Tabular regressor. Used for discrete state and action spaces.
    """
    def __init__(self, **params):
        """
        Constructor.

        # Arguments
            approximator_params (dict): parameters.
        """
        self.__name__ = 'Tabular'

        self._Q = np.zeros(params['shape'])

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used);
            y (np.array): target.
        """
        assert x.shape[1] == len(self._Q.shape), 'tabular regressor dimension ' \
                                                 'does not fit with input size.'

        idxs = [x[:, i].astype(int) for i in xrange(x.shape[1])]
        self._Q[idxs] = y

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

        idxs = [x[:, i].astype(int) for i in xrange(x.shape[1])]
        q = self._Q[idxs]

        return q

    @property
    def shape(self):
        return self._Q.shape

    def __str__(self):
        return self.__name__
