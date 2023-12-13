from mushroom_rl.core.serialization import Serializable


class GenericRegressor(Serializable):
    """
    This class is used to create a regressor that approximates a generic
    function. An arbitrary number of inputs and outputs is supported.

    """
    def __init__(self, approximator, n_inputs, **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                a generic function;
            n_inputs (int): number of inputs of the regressor;
            **params: parameters dictionary to the regressor;

        """
        self._n_inputs = n_inputs
        self.model = approximator(**params)

        self._add_save_attr(
            _n_inputs='primitive',
            model=self._get_serialization_method(approximator)
        )

    def fit(self, *z, **fit_params):
        """
        Fit the model.

        Args:
            *z: list of inputs and targets;
            **fit_params: other parameters used by the fit method of the
                regressor.

        """
        self.model.fit(*z, **fit_params)

    def predict(self, *x, **predict_params):
        """
        Predict.

        Args:
            x (list): list of inputs;
            **predict_params: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        return self.model.predict(*x, **predict_params)

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            self.model.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      ' non-parametric regressor.')

    @property
    def weights_size(self):
        return self.model.weights_size

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def diff(self, *x):
        return self.model.diff(*x)

    def __len__(self):
        return len(self.model)
