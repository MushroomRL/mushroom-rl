import numpy as np

from mushroom_rl.approximators.approximator import Approximator


class SklearnApproximator(Approximator):
    """
    Class wrapping a scikit-learn estimator as a numpy approximator.

    """

    def __init__(self, model_class, input_shape, output_shape, *, n_models=1, **model_params):
        """
        Constructor.

        Args:
            model_class (class): the scikit-learn estimator class to instantiate;
            input_shape (tuple): shape of the input;
            output_shape (tuple, list): shape of the output, or a list of shapes for estimators returning a
                tuple;
            n_models (int, 1): number of models; more than one builds an ``Ensemble``;
            **model_params: parameters passed to the constructor of the estimator.

        """
        super().__init__(input_shape, output_shape, backend='numpy')

        self._model = model_class(**model_params)
        self._fitted = False

        self._add_save_attr(_model='pickle', _fitted='primitive')

    def fit(self, x, y, **fit_params):
        """
        Fit the estimator.

        Args:
            x (np.ndarray): a batch of inputs;
            y (np.ndarray): the targets, one per input;
            **fit_params: parameters passed to the ``fit`` method of the estimator.

        """
        n_targets = int(np.prod(self._plain_output_shape))
        target_shape = () if n_targets == 1 else (n_targets,)

        x = self._flatten_inputs(np.asarray(x))
        y = np.asarray(y)
        y = y.reshape((len(y),) + target_shape)

        self._model.fit(x, y, **fit_params)
        self._fitted = True

    def predict(self, x, **predict_params):
        """
        Predict with the estimator.

        Args:
            x (np.ndarray): a single input or a batch of inputs;
            **predict_params: parameters passed to the ``predict`` method of the estimator.

        Returns:
            The prediction, or a tuple of predictions when the estimator returns one. Before the estimator is
            fitted, the prediction is zero when ``output_shape`` is a single shape.

        Raises:
            NotFittedError: if the estimator has not been fitted yet and ``output_shape`` is a list of shapes.

        """
        x = np.asarray(x)
        single = self._is_single(x)
        x = self._flatten_inputs(x[None] if single else x)

        if self._fitted or isinstance(self._output_shape, list):
            prediction = self._model.predict(x, **predict_params)
        else:
            prediction = np.zeros((len(x), *self._output_shape))

        if isinstance(prediction, tuple):
            prediction = tuple(y.reshape(len(y), *shape)
                               for y, shape in zip(prediction, self._output_shape, strict=True))
        else:
            prediction = prediction.reshape(len(prediction), *self._plain_output_shape)

        return self._unbatch(prediction, single)

    @property
    def model(self):
        """
        Returns:
            The wrapped scikit-learn estimator.

        """
        return self._model

    def _flatten_inputs(self, x):
        assert x.shape[1:] == tuple(self._input_shape)
        return x.reshape(len(x), -1)

    @property
    def _plain_output_shape(self):
        return self._output_shape[0] if isinstance(self._output_shape, list) else self._output_shape
