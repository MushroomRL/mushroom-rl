import numpy as np

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
            x (np.ndarray): inputs, a batch of samples of shape ``input_shape`` along the first axis;
            y (np.ndarray): targets, one per input, each holding ``prod(output_shape)`` values;
            **fit_params: other parameters passed to the ``fit`` method of the estimator.

        """
        x = x.reshape(-1, int(np.prod(self._input_shape)))
        if tuple(self._output_shape) == (1,):
            y = y.reshape(len(x))
        else:
            y = y.reshape(len(x), int(np.prod(self._output_shape)))

        self._model.fit(x, y, **fit_params)

    def predict(self, x, **predict_params):
        """
        Predict with the estimator.

        Args:
            x (np.ndarray): inputs, a single sample of shape ``input_shape`` or a batch of them along the first
                axis;
            **predict_params: other parameters passed to the ``predict`` method of the estimator.

        Returns:
            The predictions: an array of shape ``output_shape`` for a single sample; for a batch of ``n`` samples,
            an array of shape ``(n, *output_shape)``, or ``(n,)`` when ``output_shape`` is ``(1,)``.

        Raises:
            NotFittedError: if the estimator has not been fitted yet.

        """
        batch = x.reshape(-1, int(np.prod(self._input_shape)))
        y = self._model.predict(batch, **predict_params).reshape(len(batch), *self._output_shape)

        if x.ndim == len(self._input_shape):
            return y[0]
        elif tuple(self._output_shape) == (1,):
            return y[:, 0]
        else:
            return y

    @property
    def model(self):
        """
        Returns:
            The wrapped scikit-learn estimator.

        """
        return self._model
