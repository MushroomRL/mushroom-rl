class GenericRegressor:
    """
    This class is used to create a regressor that approximates a generic
    function. An arbitrary number of inputs and outputs is supported.

    """
    def __init__(self, approximator, n_inputs, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                a generic function;
            n_inputs (int): number of inputs of the regressor;
            **params (dict): parameters dictionary to the regressor;

        """
        self._n_inputs = n_inputs
        self._input_preprocessor = params.pop('input_preprocessor', list())
        self._output_preprocessor = params.pop('output_preprocessor', list())
        self.model = approximator(**params)

    def fit(self, *z, **fit_params):
        """
        Fit the model.

        Args:
            *z (list): list of inputs and targets;
            **fit_params (dict): other parameters used by the fit method of the
                regressor.

        """
        z = self._preprocess(*z)
        self.model.fit(*z, **fit_params)

    def predict(self, *x, **predict_params):
        """
        Predict.

        Args:
            x (list): list of inputs;
            **predict_params (dict): other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        x = self._preprocess(*x)

        return self.model.predict(*x, **predict_params)

    @property
    def weights_size(self):
        return self.model.weights_size

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def diff(self, *x):
        x = self._preprocess(*x)

        return self.model.diff(*x)

    def _preprocess(self, *z):
        x = list(z[:self._n_inputs])
        y = list(z[self._n_inputs:])

        if len(self._input_preprocessor) > 0 and not isinstance(
                self._input_preprocessor[0], list):
            self._input_preprocessor = [self._input_preprocessor]
        for i, ip in enumerate(self._input_preprocessor):
            for p in ip:
                x[i] = p(x[i])
        z = [i for i in x]

        if len(y) > 0:
            if len(self._output_preprocessor) > 0 and not isinstance(
                    self._output_preprocessor[0], list):
                self._output_preprocessor = [self._output_preprocessor]
            for o, op in enumerate(self._output_preprocessor):
                for p in op:
                    y[o] = p(y[o])
        z += [i for i in y]

        return z

    def __len__(self):
        return len(self.model)

    def __str__(self):
        return 'GenericRegressor of ' + str(self.model) + '.'
