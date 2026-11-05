import numpy as np

from mushroom_rl.approximators.approximator import Approximator


class SklearnApproximator(Approximator):
    """
    Class wrapping a scikit-learn estimator as a numpy approximator.

    """

    def __init__(self, model_class, input_shape, output_shape=(1,), n_models=1, **model_params):
        """
        Constructor.

        Args:
            model_class (class): the scikit-learn estimator class to instantiate;
            input_shape (tuple): shape of the input;
            output_shape (tuple, list, (1,)): shape of the output, or a list of shapes, one per element of the tuple
                the estimator returns with some ``predict_params`` (e.g. ``return_std=True``);
            n_models (int, 1): number of models; with more than one, an ``Ensemble`` of them is built;
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
            y (np.ndarray): targets, one per input, each holding ``prod(output_shape)`` values, or
                ``prod(output_shape[0])`` when ``output_shape`` is a list;
            **fit_params: other parameters passed to the ``fit`` method of the estimator.

        """
        output_shape = self._output_shape[0] if isinstance(self._output_shape, list) else self._output_shape
        target_shape = () if tuple(output_shape) == (1,) else (int(np.prod(output_shape)),)

        x = np.asarray(x)
        x = x.reshape(len(x), int(np.prod(self._input_shape)))
        y = np.asarray(y)
        y = y.reshape((len(y),) + target_shape)

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
            an array of shape ``(n, *output_shape)``, or ``(n,)`` when ``output_shape`` is ``(1,)``. When
            ``output_shape`` is a list, a tuple with one such array per shape in the list.

        Raises:
            NotFittedError: if the estimator has not been fitted yet.

        """
        x = np.asarray(x)
        single = x.ndim == len(self._input_shape)
        x = x.reshape(1 if single else len(x), int(np.prod(self._input_shape)))

        prediction = self._model.predict(x, **predict_params)

        if isinstance(prediction, tuple):
            outputs = list()
            for y, shape in zip(prediction, self._output_shape, strict=True):
                outputs.append(self._parse_output(y, shape, single))

            return tuple(outputs)
        else:
            return self._parse_output(prediction, self._output_shape, single)

    @property
    def model(self):
        """
        Returns:
            The wrapped scikit-learn estimator.

        """
        return self._model

    @staticmethod
    def _parse_output(y, shape, single):
        y = y.reshape(len(y), *shape)

        if single:
            return y[0]
        elif tuple(shape) == (1,):
            return y[:, 0]
        else:
            return y
