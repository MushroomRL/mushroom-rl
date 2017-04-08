from sklearn.ensemble import ExtraTreesRegressor


class ExtraTrees(object):
    """
    Wrapper class of Scikit-Learn's Extra Trees.
    """
    def __init__(self, **approximator_params):
        """
        Constructor.

        # Arguments
            approximator_params (dict): parameters.
        """
        self.__name__ = 'ExtraTrees'

        self.approximator_params = approximator_params

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).
            y (np.array): target.
            fit_params (dict): other parameters.
        """
        if not hasattr(self, 'model'):
            self.model = self._initialize()

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        predictions = self.model.predict(x)

        return predictions.ravel()

    def _initialize(self):
        return ExtraTreesRegressor(**self.approximator_params)

    def __str__(self):
        return self.__name__
