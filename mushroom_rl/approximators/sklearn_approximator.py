from mushroom_rl.approximators.approximator import Approximator


class SklearnApproximator(Approximator):
    """
    Class wrapping a scikit-learn estimator as a numpy approximator.

    """

    def __init__(self, model_class, input_shape, output_shape=(1,), **model_params):
        """
        Constructor.

        Args:
            model_class (class): the scikit-learn estimator class to instantiate;
            input_shape (tuple): shape of the input;
            output_shape (tuple, (1,)): shape of the output;
            **model_params: parameters passed to the constructor of the estimator.

        """
        super().__init__(input_shape, output_shape, backend='numpy')

        self._model = model_class(**model_params)

        self._add_save_attr(_model='pickle')

    def fit(self, x, y, **fit_params):
        """
        Fit the estimator.

        Args:
            x (np.ndarray): inputs;
            y (np.ndarray): targets;
            **fit_params: other parameters passed to the ``fit`` method of the estimator.

        """
        self._model.fit(x, y, **fit_params)

    def predict(self, x, **predict_params):
        """
        Predict with the estimator.

        Args:
            x (np.ndarray): inputs;
            **predict_params: other parameters passed to the ``predict`` method of the estimator.

        Returns:
            The predictions of the estimator.

        Raises:
            NotFittedError: if the estimator has not been fitted yet.

        """
        return self._model.predict(x, **predict_params)

    @property
    def model(self):
        """
        Returns:
            The wrapped scikit-learn estimator.

        """
        return self._model
