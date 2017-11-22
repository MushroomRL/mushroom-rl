class SimpleRegressor:
    """
    This class is used to create a regressor that approximates a generic
    function, not only an action-value function.

    """
    def __init__(self, approximator, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                a generic function;
            **params (dict): parameters dictionary to the regressor;

        """
        self._input_preprocessor = params.pop('input_preprocessor', list())
        self._output_preprocessor = params.pop('output_preprocessor', list())
        self.model = approximator(**params)

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (np.ndarray): input;
            y (np.ndarray): target;
            **fit_params (dict): other parameters used by the fit method of the
                regressor.

        """
        x, y = self._preprocess(x, y)
        self.model.fit(x, y, **fit_params)

    def predict(self, x, **predict_params):
        """
        Predict.

        Args:
            x (np.ndarray): input;
            **predict_params (dict): other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        x = self._preprocess(x)

        return self.model.predict(x, **predict_params)

    @property
    def weights_size(self):
        return self.model.weights_size

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def diff(self, x):
        x = self._preprocess(x)
        return self.model.diff(x)

    def _preprocess(self, x, y=None):
        for p in self._input_preprocessor:
            x = p(x)

        if y is not None:
            for p in self._output_preprocessor:
                y = p(y)

            return x, y
        return x

    def __len__(self):
        return len(self.model)

    def __str__(self):
        return 'SimpleRegressor of ' + str(self.model) + '.'
