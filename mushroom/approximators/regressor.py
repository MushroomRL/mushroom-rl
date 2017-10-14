from .implementations.action_regressor import ActionRegressor
from .implementations.ensemble import Ensemble
from .implementations.q_regressor import QRegressor
from .implementations.simple_regressor import SimpleRegressor


class Regressor:
    """
    This class implements the function to manage a function approximator. This
    class selects the appropriate kind of regressor to implement according to
    the parameters provided by the user; this makes this class the only one to
    use for each kind of task that has to be performed.
    The inference of the implementation to choose is done checking the provided
    values of parameters `n_actions`.
    If `n_actions` is provided, if its value is equal to
    the `output_shape` then a `QRegressor` is created, else (`output_shape`
    should be (1,)) an `ActionRegressor` is created.
    Else a `SimpleRegressor` is created.
    An `Ensemble` model can be used for all the previous implementations
    listed before simply providing a `n_models` parameter greater than 1.

    """
    def __init__(self, approximator, input_shape, output_shape=(1,),
                 n_actions=None, n_models=1, **params):
        approximator_params = params.get('params', dict())

        self._input_shape = input_shape
        self._output_shape = output_shape

        assert n_models >= 1
        self._n_models = n_models

        if n_actions is not None:
            assert len(self._output_shape) == 1 and n_actions >= 2
            if n_actions == self._output_shape[0]:
                self._impl = QRegressor(approximator, n_actions,
                                        approximator_params, **params)
            else:
                self._impl = ActionRegressor(approximator, n_actions,
                                             approximator_params, **params)
        else:
            self._impl = SimpleRegressor(approximator, approximator_params,
                                         **params)

        if self._n_models > 1:
            prediction = params.get('prediction', 'mean')
            self._impl = Ensemble(self._impl, self._n_models, prediction)

    def fit(self, *z, **fit_params):
        self._impl.fit(*z, **fit_params)

    def predict(self, *z, **predict_params):
        return self._impl.predict(*z, **predict_params)

    @property
    def model(self):
        return self._impl.model

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def get_weights(self):
        try:
            return self._impl.get_weights()
        except AttributeError:
            raise NotImplementedError('Attempt to get weights of a'
                                      'non-parametric regressor.')

    def set_weights(self, w):
        try:
            self._impl.set_weights(w)
        except AttributeError:
            raise NotImplementedError('Attempt to set weights of a'
                                      'non-parametric regressor.')

    def diff(self, *z):
        try:
            return self._impl.diff(*z)
        except AttributeError:
            raise NotImplementedError('Attempt to compute derivative of a'
                                      'non-differentiable regressor.')

    def __getitem__(self, idx):
        return self._impl[idx]

    def __str__(self):
        return str(self._impl)
