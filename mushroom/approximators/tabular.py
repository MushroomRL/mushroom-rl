import numpy as np


class Tabular(object):
    """
    Tabular regressor. Used for discrete state and action spaces.

    """
    def __init__(self, shape, initial_value=0., **params):
        """
        Constructor.

        Args:
            shape (tuple): the shape of the tabular regressor.
            initial_value (float): the initial value for each entry of the
                tabular regressor.

        """
        self.__name__ = 'Tabular'

        self.Q = np.ones(shape) * initial_value

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters.

        """
        assert x.shape[1] == len(self.Q.shape), 'tabular regressor dimension ' \
                                                 'does not fit with input size.'

        idxs = [x[:, i].astype(int) for i in xrange(x.shape[1])]
        self.Q[idxs] = y

    def predict(self, x):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions;

        Returns:
            The predictions of the model.

        """
        assert x.shape[1] == len(self.Q.shape), 'tabular regressor dimension ' \
                                                 'does not fit with input size.'

        idxs = [x[:, i].astype(int) for i in xrange(x.shape[1])]
        q = self.Q[idxs]

        return q

    @property
    def shape(self):
        return self.Q.shape

    def __str__(self):
        return self.__name__
