class SimpleRegressor:
    def __init__(self, approximator, approximator_params, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                a generic function;
            approximator_params (dict): parameters dictionary to the regressor;
            **params (dict): parameters dictionary used by the
                `SimpleRegressor`.

        """
        self.model = approximator(**approximator_params)
        self._input_preprocessor = params.get('input_preprocessor', list())
        self._output_preprocessor = params.get('output_preprocessor', list())

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (np.array): input;
            y (np.array): target;
            **fit_params (dict): other parameters used by the fit method of the
                regressor.

        """
        x, y = self._preprocess(x, y)
        self.model.fit(x, y, **fit_params)

    def predict(self, x, **predict_params):
        """
        Predict.

        Args:
            x (np.array): input;
            **predict_params (dict): other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        x = self._preprocess(x)

        return self.model.predict(x, **predict_params)

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

    def __str__(self):
        return 'SimpleRegressor of ' + str(self.model) + '.'
