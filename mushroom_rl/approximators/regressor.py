import numpy as np

from mushroom_rl.core.serialization import Serializable
from .ensemble import Ensemble
from ._implementations.q_regressor import QRegressor
from ._implementations.action_regressor import ActionRegressor
from ._implementations.generic_regressor import GenericRegressor


class Regressor(Serializable):
    """
    This class implements the function to manage a function approximator. This
    class selects the appropriate kind of regressor to implement according to
    the parameters provided by the user; this makes this class the only one to
    use for each kind of task that has to be performed.
    The inference of the implementation to choose is done checking the provided
    values of parameters ``n_actions``.
    If ``n_actions`` is provided, it means that the user wants to implement an
    approximator of the Q-function: if the value of ``n_actions`` is equal to
    the ``output_shape`` then a ``QRegressor`` is created, else
    (``output_shape`` should be (1,)) an ``ActionRegressor`` is created.
    Otherwise a ``GenericRegressor`` is created.
    An ``Ensemble`` model can be used for all the previous implementations
    listed before simply providing a ``n_models`` parameter greater than 1.

    """
    def __init__(self, approximator, input_shape, output_shape=None,
                 n_actions=None, n_models=None, **params):
        """
        Constructor.

        Args:
            approximator (class): the approximator class to use to create
                the model;
            input_shape (tuple): the shape of the input of the model;
            output_shape (tuple, None): the shape of the output of the model;
            n_actions (int, None): number of actions considered to create a
                ``QRegressor`` or an ``ActionRegressor``;
            n_models (int, 1): number of models to create;
            **params: other parameters to create each model.

        """
        if n_actions is None:
            assert output_shape is not None
        else:
            if output_shape is None:
                output_shape = (1,)

        if not approximator.__module__.startswith('sklearn'):
            params['input_shape'] = input_shape
            params['output_shape'] = output_shape

        self._input_shape = input_shape
        self._output_shape = output_shape

        self.n_actions = n_actions
        self._n_models = n_models
        self._logger = None

        if self._n_models is not None:
            assert self._n_models >= 1
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
            self._impl = GenericRegressor(approximator,
                                          len(self.input_shape),
                                          **params)

        self._logger = None
        self._loss_filename = None

        self._add_save_attr(
            _input_shape='primitive',
            _output_shape='primitive',
            n_actions='primitive',
            _n_models='primitive',
            _impl='mushroom',
            _logger='none'
        )

    def __call__(self, *z, **predict_params):
        return self.predict(*z, **predict_params)

    def _ndim(self):
        if isinstance(self._input_shape[0], tuple):
            ndim = len(self._input_shape[0])
        else:
            ndim = len(self._input_shape)

        return ndim

    def fit(self, *z, **fit_params):
        """
        Fit the model.

        Args:
            *z: list of input of the model;
            **fit_params: parameters to use to fit the model.

        """
        ndim = self._ndim()

        if z[0].ndim == ndim:
            z = [np.expand_dims(z_i, axis=0) for z_i in z]
        self._impl.fit(*z, **fit_params)

        self._log()

    def predict(self, *z, **predict_params):
        """
        Predict the output of the model given an input.

        Args:
            *z: list of input of the model;
            **predict_params: parameters to use to predict with the model.

        Returns:
            The model prediction.

        """
        ndim = self._ndim()

        if z[0].ndim == ndim:
            z = [np.expand_dims(z_i, axis=0) if isinstance(z_i, np.ndarray)
                 else z_i.unsqueeze(0)
                 for z_i in z]

            return self._impl.predict(*z, **predict_params)[0]
        else:
            return self._impl.predict(*z, **predict_params)

    @property
    def model(self):
        """
        Returns:
             The model object.

        """
        return self._impl.model

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            self._impl.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      'non-parametric regressor.')

    @property
    def input_shape(self):
        """
        Returns:
             The shape of the input of the model.

        """
        return self._input_shape

    @property
    def output_shape(self):
        """
        Returns:
             The shape of the output of the model.

        """
        return self._output_shape

    @property
    def weights_size(self):
        """
        Returns:
             The shape of the weights of the model.

        """
        try:
            return self._impl.weights_size
        except AttributeError:
            raise NotImplementedError('Attempt to get shape of weights of a'
                                      ' non-parametric regressor.')

    def get_weights(self):
        """
        Returns:
             The weights of the model.

        """
        try:
            return self._impl.get_weights()
        except AttributeError:
            raise NotImplementedError('Attempt to get weights of a'
                                      ' non-parametric regressor.')

    def set_weights(self, w):
        """
        Args:
            w (list): list of weights to be set in the model.

        """
        try:
            self._impl.set_weights(w)
        except AttributeError:
            raise NotImplementedError('Attempt to set weights of a'
                                      ' non-parametric regressor.')

    def diff(self, *z):
        """
        Args:
            *z: the input of the model.

        Returns:
             The derivative of the model.

        """
        try:
            return self._impl.diff(*z)
        except AttributeError:
            raise NotImplementedError('Attempt to compute derivative of a'
                                      ' non-differentiable regressor.')

    def __len__(self):
        return 1 if self._n_models is None else len(self._impl)

    def __getitem__(self, item):
        if self._n_models is None:
            return self.model
        else:
            return self.model[item]

    def set_logger(self, logger, loss_filename=None):
        """
        Setter that can be used to pass a logger to the regressor.

        Args:
            logger (Logger): the logger to be used by the regressor;
            loss_filename (str, None): optional string to specify the loss filename.

        """
        self._logger = logger
        self._loss_filename = loss_filename

    def _log(self):
        if self._logger:
            losses_fit = list()
            for i in range(len(self)):
                model_i = self[i]
                if hasattr(model_i, 'loss_fit'):
                    m_loss = model_i.loss_fit
                    if hasattr(m_loss, 'squeeze'):
                        m_loss = m_loss.squeeze()
                    losses_fit.append(m_loss)

            losses_fit = np.array(losses_fit).squeeze()

            if self._loss_filename is None:
                save_dict = dict(loss=losses_fit)
            else:
                save_dict = {self._loss_filename: losses_fit}

            self._logger.log_numpy(**save_dict)


