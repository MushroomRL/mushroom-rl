from mushroom_rl.approximators.approximator import Approximator, Ensemble


class QApproximator(Approximator):
    """
    Interface for Q-function approximators. Selects the appropriate concrete subclass based on
    the construction arguments:

    - ``n_models > 1``: ``QApproximatorEnsemble`` — ensemble of independent Q-approximators;
    - ``output_shape[0] != n_actions``: ``QApproximatorAction`` — one model per action;
    - ``output_shape[0] == n_actions``: ``QApproximatorSimple`` — single multi-output model.

    """

    def __new__(cls, approximator=None, n_actions=None, output_shape=(1,), n_models=1, **kwargs):
        if cls is not QApproximator:
            return object.__new__(cls)
        if n_models > 1:
            return object.__new__(QApproximatorEnsemble)
        elif output_shape[0] != n_actions:
            return object.__new__(QApproximatorAction)
        else:
            return object.__new__(QApproximatorSimple)

    @property
    def n_actions(self):
        return self._n_actions


class QApproximatorSimple(QApproximator):
    """
    Approximates the Q-function with a single multi-output model where each output corresponds
    to the Q-value of one action. Used when ``output_shape[0] == n_actions``.

    """

    def __init__(self, approximator, n_actions, output_shape=(1,), n_models=1, input_shape=None,
                 **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the Q-function;
            n_actions (int): number of actions;
            output_shape (tuple, (1,)): shape of the output of the model;
            input_shape (tuple, None): shape of the input of the model;
            **params: other parameters passed to the approximator.

        """
        assert len(output_shape) == 1 and n_actions >= 2
        if input_shape is not None:
            params['input_shape'] = input_shape
        params['output_shape'] = output_shape
        model = approximator(**params)
        backend = getattr(model, '_backend', None)
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                         backend=backend.get_backend_name() if backend is not None else 'numpy')
        self._n_actions = n_actions
        self._models = [model]
        self._add_save_attr(
            _n_actions='primitive',
            _models=self._get_serialization_method(approximator)
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
                to get the Q-value of the selected action;
            **predict_params: other parameters passed to the model's predict method.

        Returns:
            The Q-value predictions.

        """
        assert len(z) == 1 or len(z) == 2
        state = z[0]
        q = self._models[0].predict(state, **predict_params)
        if len(z) == 2:
            action = z[1].ravel()
            if q.ndim == 1:
                return q[action]
            return q[self._backend.arange(0, q.shape[0]), action]
        return q

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
        return self._models[0].diff(state, action).squeeze()

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            self._models[0].reset()
        except AttributeError:
            raise NotImplementedError


class QApproximatorAction(QApproximator):
    """
    Approximates the Q-function with one independent model per action. Used when
    ``output_shape[0] != n_actions``, typically in MDPs with discrete actions where a
    separate approximator is trained for each action.

    """

    def __init__(self, approximator, n_actions, output_shape=(1,), n_models=1, input_shape=None,
                 **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the Q-function of each action;
            n_actions (int): number of actions, determines the number of models created;
            output_shape (tuple, (1,)): shape of the output of each model;
            input_shape (tuple, None): shape of the input of each model;
            **params: other parameters passed to each model.

        """
        assert n_actions >= 2
        is_sklearn = approximator.__module__.startswith('sklearn')
        if input_shape is not None and not is_sklearn:
            params['input_shape'] = input_shape
        self._n_actions = n_actions
        self._models = [approximator(**params) for _ in range(n_actions)]
        backend = getattr(self._models[0], '_backend', None)
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                         backend=backend.get_backend_name() if backend is not None else 'numpy')
        self._add_save_attr(
            _n_actions='primitive',
            _models=self._get_serialization_method(approximator)
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
                to get the Q-value of the selected action;
            **predict_params: other parameters passed to each model's predict method.

        Returns:
            The Q-value predictions.

        """
        assert len(z) == 1 or len(z) == 2
        state = self._backend.atleast_2d(z[0])
        if len(z) == 2:
            action = self._backend.atleast_2d(z[1])
            q = self._backend.zeros(state.shape[0])
            for i in range(self._n_actions):
                mask = (action == i)[:, 0]
                if mask.any():
                    q[mask] = self._models[i].predict(state[mask], **predict_params).squeeze()
        else:
            q = self._backend.zeros(state.shape[0], self._n_actions)
            for i in range(self._n_actions):
                q[:, i] = self._models[i].predict(state, **predict_params).squeeze()
        return self._backend.squeeze(q)

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
            A list of per-action gradients when ``action`` is ``None``, or a single
            zero-padded gradient vector otherwise.

        """
        if action is None:
            return [m.diff(state) for m in self._models]
        a = action[0]
        s = self._models[0].weights_size
        diff = self._backend.zeros(s * self._n_actions, dtype=float)
        diff[s * a:s * (a + 1)] = self._models[a].diff(state)
        return diff

    def reset(self):
        """
        Reset the parameters of all action models.

        """
        try:
            for m in self._models:
                m.reset()
        except AttributeError:
            raise NotImplementedError


class QApproximatorEnsemble(QApproximator, Ensemble):
    """
    Ensemble of ``QApproximator`` models. Each model is an independent ``QApproximatorSimple``
    or ``QApproximatorAction`` depending on the output shape.

    """

    def __init__(self, approximator, n_actions, output_shape=(1,), n_models=1, prediction='mean', **params):
        """
        Constructor.

        Args:
            approximator (class): the model class for each ensemble member;
            n_actions (int): number of actions;
            output_shape (tuple, (1,)): shape of the output of each model;
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): aggregation mode across models.
                One of ``'mean'``, ``'sum'``, ``'min'``, ``'max'``;
            **params: other parameters passed to each model.

        """
        assert n_actions >= 2 and n_models > 1
        Ensemble.__init__(self, QApproximator, n_models, prediction=prediction,
                          approximator=approximator, n_actions=n_actions,
                          output_shape=output_shape, **params)
        backend = getattr(self._models[0], '_backend', None)
        if backend is not None:
            self._backend = backend
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