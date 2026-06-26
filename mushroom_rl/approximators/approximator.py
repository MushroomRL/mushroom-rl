from sklearn.exceptions import NotFittedError

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.serialization import Serializable


class Approximator(Serializable):
    """
    Base class for all approximators. Handles logger attachment and dispatches
    to an ``Ensemble`` when ``n_models > 1`` is requested at construction time.

    """

    def __new__(cls, *args, n_models=1, **kwargs):
        if issubclass(cls, Ensemble):
            return Serializable.__new__(cls)
        if n_models > 1:
            ensemble = Serializable.__new__(Ensemble)
            Ensemble.__init__(ensemble, cls, n_models, **kwargs)
            return ensemble
        return Serializable.__new__(cls)

    def __init__(self, input_shape, output_shape, backend='numpy'):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._backend = ArrayBackend.get_array_backend(backend)
        self._add_save_attr(_input_shape='primitive', _output_shape='primitive',
                            _backend='primitive')

    def fit(self, *args, **kwargs):
        """
        Fit the model on the provided data.

        Args:
            *args: list of inputs;
            **kwargs: other parameters used by the fit method of the regressor.

        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        Predict the output of the model given an input.

        Args:
            *args: list of inputs;
            **kwargs: other parameters used by the predict method of the regressor.

        Returns:
            The model prediction.

        """
        raise NotImplementedError

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self

    def __call__(self, *z, **kw):
        return self.predict(*z, **kw)

    @property
    def input_shape(self):
        """
        Returns:
            The shape of the input of the approximator.

        """
        return self._input_shape

    @property
    def output_shape(self):
        """
        Returns:
            The shape of the output of the approximator.

        """
        return self._output_shape

    def _log(self):
        if self._logger:
            if not hasattr(self, 'loss_fit'):
                return
            loss = self.loss_fit
            if loss is None:
                return
            if hasattr(loss, 'squeeze'):
                loss = loss.squeeze()
            name = self._log_label or 'loss'
            self._logger.log_training(self._log_prefix, **{name: loss})


class Ensemble(Approximator):
    """
    This class is used to create an ensemble of approximators.

    """

    def __init__(self, model, n_models, prediction='mean', backend='numpy', **params):
        """
        Constructor.

        Args:
            model (class): the model class to use for each element of the ensemble;
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): the type of prediction to make across models.
                One of ``'mean'``, ``'sum'``, ``'min'``, ``'max'``;
            backend (str, 'numpy'): array backend to use;
            **params: parameters dictionary to create each model.

        """
        super().__init__(input_shape=params.get('input_shape'), output_shape=params.get('output_shape'),
                         backend=backend)

        self._prediction = prediction
        self._models = []

        for _ in range(n_models):
            self._models.append(model(**params))

        self._add_save_attr(
            _models=self._get_serialization_method(model),
            _prediction='primitive'
        )

    def __len__(self):
        return len(self._models)

    def __getitem__(self, idx):
        return self._models[idx]

    def fit(self, *z, idx=None, **fit_params):
        """
        Fit the ``idx``-th model of the ensemble if ``idx`` is provided, every model otherwise.

        Args:
            *z: list of inputs to use to fit each model;
            idx (int, None): index of the model to fit;
            **fit_params: other parameters passed to each model's fit method.

        """
        if idx is None:
            for i in range(len(self)):
                self[i].fit(*z, **fit_params)
        else:
            self[idx].fit(*z, **fit_params)

    def predict(self, *z, idx=None, prediction=None, compute_variance=False, **predict_params):
        """
        Predict.

        Args:
            *z: list of inputs to use to predict with each model;
            idx (int, None): index of the model to use for prediction. If ``None``, all models
                are used and aggregated according to ``prediction``;
            prediction (str, None): aggregation mode, overrides the constructor default.
                One of ``'mean'``, ``'sum'``, ``'min'``, ``'max'``, or ``None`` to return
                all predictions stacked along axis 0;
            compute_variance (bool, False): if ``True``, also return the variance across models;
            **predict_params: other parameters passed to each model's predict method.

        Returns:
            The stacked predictions along axis 0 if ``prediction`` is ``None``, the aggregated
            predictions otherwise, or a list ``[predictions, variance]`` if
            ``compute_variance`` is ``True``.

        """
        if idx is None:
            idx = list(range(len(self)))

        if isinstance(idx, int):
            try:
                results = self[idx].predict(*z, **predict_params)
            except NotFittedError:
                raise NotFittedError
        else:
            predictions = list()
            for i in idx:
                try:
                    predictions.append(self[i].predict(*z, **predict_params))
                except NotFittedError:
                    raise NotFittedError

            prediction = prediction if prediction is not None else self._prediction
            predictions = self._backend.stack(predictions, 0)

            if prediction is None:
                return predictions
            elif prediction == 'mean':
                results = predictions.mean(0)
            elif prediction == 'sum':
                results = predictions.sum(0)
            elif prediction == 'min':
                results = self._backend.min(predictions, 0)
            elif prediction == 'max':
                results = self._backend.max(predictions, 0)
            else:
                raise ValueError

            if compute_variance:
                results = [results, predictions.var(0)]

        return results

    def set_logger(self, logger, prefix=None, label=None):
        """
        Attach the logger to each model of the ensemble so that every model logs its own loss during
        its ``fit``. The model index is appended to the metric name (e.g. ``critic/loss_0``).

        Args:
            logger (Logger): the logger object;
            prefix (str, None): optional group prepended to the logged metric names;
            label (str, None): optional name used for the loss. Defaults to ``'loss'``.

        """
        super().set_logger(logger, prefix, label)
        name = label or 'loss'
        for i, m in enumerate(self._models):
            m.set_logger(logger, prefix, f'{name}_{i}')

    @property
    def n_actions(self):
        """
        Returns:
            The number of actions of the first model in the ensemble.

        """
        return self._models[0].n_actions

    def reset(self):
        """
        Reset the parameters of all models in the ensemble.

        """
        try:
            for m in self._models:
                m.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a non-parametric regressor.')