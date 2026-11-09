from mushroom_rl.core import MushroomObject
from mushroom_rl.approximators.approximator import Approximator, Ensemble


class QApproximator(Approximator):
    """
    Interface for Q-function approximators. Selects the appropriate concrete subclass based on
    the construction arguments:

    - ``n_models > 1``: ``QApproximatorEnsemble`` — ensemble of independent Q-approximators;
    - ``output_shape == ()``: ``QApproximatorAction`` — one scalar model per action;
    - ``output_shape == (n_actions,)``: ``QApproximatorSimple`` — single multi-output model.

    """

    def __new__(cls, approximator=None, n_actions=None, input_shape=None, output_shape=(), n_models=1, **kwargs):
        if cls is not QApproximator:
            return MushroomObject.__new__(cls)
        if n_models > 1:
            return MushroomObject.__new__(QApproximatorEnsemble)
        elif tuple(output_shape) == ():
            return MushroomObject.__new__(QApproximatorAction)
        elif tuple(output_shape) == (n_actions,):
            return MushroomObject.__new__(QApproximatorSimple)
        else:
            raise ValueError(f'output_shape must be () for one model per action or ({n_actions},) for a single '
                             f'model, got {output_shape}.')

    @property
    def n_actions(self):
        return self._n_actions


class QApproximatorSimple(QApproximator):
    """
    Approximates the Q-function with a single multi-output model where each output corresponds
    to the Q-value of one action. Used when ``output_shape == (n_actions,)``.

    """

    def __init__(self, approximator, n_actions, input_shape, output_shape, n_models=1,
                 **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the Q-function;
            n_actions (int): number of actions;
            input_shape (tuple): shape of the input of the model;
            output_shape (tuple): shape of the output of the model;
            **params: other parameters passed to the approximator.

        """
        assert len(output_shape) == 1 and n_actions >= 2
        params['input_shape'] = input_shape
        params['output_shape'] = output_shape
        model = approximator(**params)
        super().__init__(input_shape=input_shape, output_shape=output_shape, backend=model.backend_name,
                         device=model.device)
        self._n_actions = n_actions
        self._models = [model]
        self._add_save_attr(
            _n_actions='primitive',
            _models='mushroom'
        )

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model.

        Args:
            state: states;
            action: actions;
            q: target Q-values;
            **fit_params: other parameters passed to the model's fit method.

        """
        self._models[0].fit(state, action, q, **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z: either ``(state,)`` to get all Q-values, or ``(state, action)``
                to get the Q-value of the selected action. A state with exactly ``len(input_shape)``
                dimensions is treated as a single sample;
            **predict_params: other parameters passed to the model's predict method.

        Returns:
            The Q-value predictions.

        """
        assert len(z) == 1 or len(z) == 2
        single = self._is_single(z[0])
        state = self._backend.expand_dims(z[0], 0) if single else z[0]
        q = self._models[0].predict(state, **predict_params)
        if len(z) == 2:
            q = q[self._backend.arange(0, q.shape[0]), z[1].ravel()]

        return self._unbatch(q, single)

    @property
    def model(self):
        """
        Returns:
            The underlying model.

        """
        return self._models[0]

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._models[0].weights_size

    def get_weights(self):
        """
        Returns:
            The set of weights of the model.

        """
        return self._models[0].get_weights()

    def set_weights(self, w):
        """
        Setter.

        Args:
            w: the set of weights to set.

        """
        self._models[0].set_weights(w)

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. the model parameters.

        Args:
            state: the state input;
            action (int, None): if provided, return the derivative for that action only.

        Returns:
            The gradient of the Q-value w.r.t. the model parameters.

        """
        if action is None:
            return self._models[0].diff(state)
        return self._models[0].diff(state, action)


class QApproximatorAction(QApproximator):
    """
    Approximates the Q-function with one independent model per action. Used when
    ``output_shape == ()``, typically in MDPs with discrete actions where a
    separate approximator is trained for each action.

    """

    def __init__(self, approximator, n_actions, input_shape, output_shape=(), n_models=1,
                 **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the Q-function of each action;
            n_actions (int): number of actions, determines the number of models created;
            input_shape (tuple): shape of the input of each model;
            output_shape (tuple, ()): shape of the output of each model;
            **params: other parameters passed to each model.

        """
        assert n_actions >= 2
        params['input_shape'] = input_shape
        params['output_shape'] = output_shape
        self._n_actions = n_actions
        self._models = [approximator(**params) for _ in range(n_actions)]
        super().__init__(input_shape=input_shape, output_shape=(n_actions,), backend=self._models[0].backend_name,
                         device=self._models[0].device)
        self._add_save_attr(
            _n_actions='primitive',
            _models='mushroom'
        )

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model for each action using only the samples corresponding to that action.

        Args:
            state: states;
            action: actions;
            q: target Q-values;
            **fit_params: other parameters passed to each model's fit method.

        """
        for i in range(self._n_actions):
            mask = (action == i)[:, 0]
            if mask.any():
                self._models[i].fit(state[mask], q[mask], **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z: either ``(state,)`` to get all Q-values, or ``(state, action)``
                to get the Q-value of the selected action. A state with exactly ``len(input_shape)``
                dimensions is treated as a single sample;
            **predict_params: other parameters passed to each model's predict method.

        Returns:
            The Q-value predictions.

        """
        assert len(z) == 1 or len(z) == 2
        single = self._is_single(z[0])
        state = self._backend.expand_dims(z[0], 0) if single else z[0]
        if len(z) == 2:
            action = self._backend.atleast_2d(z[1])
            q = self._backend.zeros(state.shape[0], device=self._device)
            for i in range(self._n_actions):
                mask = (action == i)[:, 0]
                if mask.any():
                    q[mask] = self._models[i].predict(state[mask], **predict_params)
        else:
            q = self._backend.zeros(state.shape[0], self._n_actions, device=self._device)
            for i in range(self._n_actions):
                q[:, i] = self._models[i].predict(state, **predict_params)

        return self._unbatch(q, single)

    @property
    def model(self):
        """
        Returns:
            The list of per-action models.

        """
        return self._models

    @property
    def weights_size(self):
        """
        Returns:
            The total size of the weights across all action models.

        """
        return self._models[0].weights_size * self._n_actions

    def get_weights(self):
        """
        Returns:
            The concatenated weights of all action models.

        """
        return self._backend.concatenate([m.get_weights() for m in self._models], 0)

    def set_weights(self, w):
        """
        Setter. Splits ``w`` evenly across action models.

        Args:
            w: the set of weights to set.

        """
        size = self._models[0].weights_size
        for i, m in enumerate(self._models):
            m.set_weights(w[i * size:(i + 1) * size])

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. the model parameters.

        Args:
            state: the state input;
            action (int, None): if provided, return a zero-padded gradient vector with the
                derivative of the selected action's model in the corresponding block.

        Returns:
            The zero-padded gradients of all actions, one column per action, when ``action`` is ``None``, or
            the single zero-padded gradient vector of ``action`` otherwise.

        """
        s = self._models[0].weights_size
        if action is None:
            diff = self._backend.zeros(s * self._n_actions, self._n_actions, dtype=float, device=self._device)
            for a, m in enumerate(self._models):
                diff[s * a:s * (a + 1), a] = m.diff(state)
            return diff
        a = action[0]
        diff = self._backend.zeros(s * self._n_actions, dtype=float, device=self._device)
        diff[s * a:s * (a + 1)] = self._models[a].diff(state)
        return diff


class QApproximatorEnsemble(QApproximator, Ensemble):
    """
    Ensemble of ``QApproximator`` models. Each model is an independent ``QApproximatorSimple``
    or ``QApproximatorAction`` depending on the output shape.

    """

    def __init__(self, approximator, n_actions, input_shape, output_shape=(), n_models=1,
                 prediction='mean', **params):
        """
        Constructor.

        Args:
            approximator (class): the model class for each ensemble member;
            n_actions (int): number of actions;
            input_shape (tuple): shape of the input of each model;
            output_shape (tuple, ()): shape of the output of each model;
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): aggregation mode across models.
                One of ``'mean'``, ``'sum'``, ``'min'``, ``'max'``;
            **params: other parameters passed to each model.

        """
        assert n_actions >= 2 and n_models > 1
        Ensemble.__init__(self, QApproximator, n_models, prediction=prediction,
                          approximator=approximator, n_actions=n_actions,
                          input_shape=input_shape, output_shape=output_shape, **params)
        self._n_actions = n_actions
        self._add_save_attr(_n_actions='primitive')

    @property
    def weights_size(self):
        """
        Returns:
            The shape of the stacked weights matrix ``(n_models, weights_size_per_model)``.

        """
        return len(self._models), self._models[0].weights_size

    def get_weights(self):
        """
        Returns:
            The stacked weights of all models, shape ``(n_models, weights_size_per_model)``.

        """
        return self._backend.stack([m.get_weights() for m in self._models], 0)

    def set_weights(self, w):
        """
        Set weights for each model in the ensemble independently.

        Args:
            w: stacked weights of shape ``(n_models, weights_size_per_model)``.

        """
        for i, m in enumerate(self._models):
            m.set_weights(w[i])

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. the model parameters for each model, stacked.

        Args:
            state: the state input;
            action (int, None): if provided, return gradients for that action only.

        Returns:
            The stacked derivatives of all models w.r.t. the model parameters.

        """
        return self._backend.stack([m.diff(state, action) for m in self._models], 0)
