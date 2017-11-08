from ._implementations.q_regressor import QRegressor
from ._implementations.action_regressor import ActionRegressor
from ._implementations.ensemble import Ensemble
from ._implementations.simple_regressor import SimpleRegressor


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
    An `Ensemble` model can be used for all the previous _implementations
    listed before simply providing a `n_models` parameter greater than 1.

    """
    def __init__(self, approximator, input_shape, output_shape=(1,),
                 n_actions=None, n_models=1, **params):
        """
        Constructor.

        Args:
            approximator (object): the approximator class to use to create
                the model;
            input_shape (tuple): the shape of the input of the model;
            output_shape (tuple, (1,)): the shape of the output of the model;
            n_actions (int, None): number of actions considered to create a
                `QRegressor` or an `ActionRegressor`;
            n_models (int, 1): number of models to create;
            **params (dict): other parameters to create each model.

        """
        if not approximator.__module__.startswith('sklearn'):
            params['input_shape'] = input_shape
            params['output_shape'] = output_shape

        self._input_shape = input_shape
        self._output_shape = output_shape

        self.n_actions = n_actions

        assert n_models >= 1
        self._n_models = n_models

        if self._n_models > 1:
            params['model'] = approximator
            params['n_models'] = n_models
            approximator = Ensemble

        if n_actions is not None:
            assert len(self._output_shape) == 1 and n_actions >= 2
            if n_actions == self._output_shape[0]:
                self._impl = QRegressor(approximator, **params)
            else:
                self._impl = ActionRegressor(approximator, n_actions, **params)
        else:
            self._impl = SimpleRegressor(approximator, **params)

    def __call__(self, *z, **predict_params):
        return self.predict(*z, **predict_params)

    def fit(self, *z, **fit_params):
        """
        Fit the model.

        Args:
            *z (list): list of input of the model;
            **fit_params (dict): parameters to use to fit the model.

        """
        self._impl.fit(*z, **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict the output of the model given an input.

        Args:
            *z (list): list of input of the model;
            **predict_params(dict): parameters to use to predict with the model.

        """
        return self._impl.predict(*z, **predict_params)

    @property
    def model(self):
        """
        Returns:
             the model object.

        """
        return self._impl.model

    @property
    def input_shape(self):
        """
        Returns:
             the shape of the input of the model.

        """
        return self._input_shape

    @property
    def output_shape(self):
        """
        Returns:
             the shape of the output of the model.

        """
        return self._output_shape

    @property
    def weights_size(self):
        """
        Returns:
             the shape of the weights of the model.

        """
        try:
            return self._impl.weights_size
        except AttributeError:
            raise NotImplementedError('Attempt to get shape of weights of a'
                                      ' non-parametric regressor.')

    def get_weights(self):
        """
        Returns:
             the weights of the model.

        """
        try:
            return self._impl.get_weights()
        except AttributeError:
            raise NotImplementedError('Attempt to get weights of a'
                                      ' non-parametric regressor.')

    def set_weights(self, w):
        """
        Args:
            w ([list, np.array]): list of weights to be set in the model.

        """
        try:
            self._impl.set_weights(w)
        except AttributeError:
            raise NotImplementedError('Attempt to set weights of a'
                                      ' non-parametric regressor.')

    def diff(self, *z):
        """
        Args:
            *z (list): the input of the model.

        Returns:
             the derivative of the model.

        """
        try:
            return self._impl.diff(*z)
        except AttributeError:
            raise NotImplementedError('Attempt to compute derivative of a'
                                      ' non-differentiable regressor.')

    def __len__(self):
        return len(self._impl)

    def __str__(self):
        return str(self._impl)
