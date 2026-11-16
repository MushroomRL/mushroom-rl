from mushroom_rl.core import ArrayBackend, MushroomBackendObject


class Policy(MushroomBackendObject):
    """
    Interface representing a generic policy.
    A policy is a probability distribution that gives the probability of taking
    an action given a specified state.
    A policy is used by mushroom agents to interact with the environment.

    """
    def __init__(self, backend, device=None):
        """
        Constructor.

        Args:
            backend (str): name of the array backend the policy operates in;
            device (str, None): device the policy data is stored on, or ``None`` for the default one. Only allowed
                with the torch backend.

        """
        super().__init__(backend, device)

    def __call__(self, state, action):
        """
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            state: state where you want to evaluate the policy density;
            action: action where you want to evaluate the policy density.

        Returns:
            The probability of all actions following the policy in the given state if the list contains only the state,
            else the probability of the given action in the given state following the policy. If the action space is
            continuous, state and action must be provided

        """
        raise NotImplementedError

    def draw_action(self, state, **kwargs):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state: the state where the agent is;
            **kwargs: additional per-timestep conditioning inputs assembled by the agent; policies that do not
                consume them can ignore the keyword arguments.

        Returns:
            The action sampled from the policy.

        """
        raise NotImplementedError

    def draw_action_greedy(self, state, **kwargs):
        """
        Return the greedy action in ``state``, used during greedy evaluation. This is the mode
        of the policy (e.g. the mean of a Gaussian, the argmax of a softmax). Policies that do not define a
        greedy action must not override this method.

        Args:
            state: the state where the agent is;
            **kwargs: additional per-timestep conditioning inputs assembled by the agent; policies that do not
                consume them can ignore the keyword arguments.

        Returns:
            The greedy action of the policy.

        """
        raise NotImplementedError("'{}' does not define a greedy action.".format(type(self).__name__))

    def reset(self):
        """
        Useful when the policy needs a special initialization at the beginning of an episode.

        Returns:
            The initial policy state (by default None).

        """
        return None

    def reset_vectorized(self, start_mask):
        """
        Reset the policy for the environments selected by ``start_mask`` at the beginning of an episode.

        Args:
            start_mask: boolean mask selecting the environments that are starting a new episode.

        Returns:
            The initial policy states (by default None).

        """
        return None

    def stop(self):
        """
        Called at the end of a run to reset any transient internal state. No-op by default.

        """
        pass

    @property
    def is_stateful(self):
        """
        Whether the policy carries an internal state that is updated step-by-step.

        """
        return False


class StatefulPolicy(Policy):
    """
    Interface representing a stateful policy, i.e. a policy carrying a latent internal state updated at every step.

    Such a state is, for instance, the hidden state of a recurrent network, the noise of an Ornstein-Uhlenbeck
    process, or the phase of a movement primitive.

    The current policy state is stored inside the policy and exposed through the ``policy_state`` property, so that the
    :class:`~mushroom_rl.core.Core` can record it into the dataset for logging and learning. The query methods, instead,
    take the policy state explicitly and never touch the stored one.

    """
    def __init__(self, policy_state_shape, **kwargs):
        """
        Constructor.

        Args:
            policy_state_shape (tuple): the shape of the internal state of the policy;
            **kwargs: the remaining keyword arguments, passed to the constructor of the base policy.

        """
        assert isinstance(policy_state_shape, tuple)
        super().__init__(**kwargs)

        self.policy_state_shape = policy_state_shape
        self._policy_state = None

        self._add_save_attr(
            policy_state_shape='primitive',
            _policy_state='none'
        )

    @property
    def is_stateful(self):
        return True

    @property
    def policy_state(self):
        """
        The current internal state of the policy.

        """
        return self._policy_state

    @policy_state.setter
    def policy_state(self, value):
        self._policy_state = None if value is None else ArrayBackend.get_array_backend_from(value).copy(value)

    def draw_action(self, state, policy_state=None, **kwargs):
        """
        Sample an action in ``state`` using the policy.

        When ``policy_state`` is not provided, the policy uses and updates its internal state. When it is provided, the
        policy uses the given state and leaves the internal one untouched (functional evaluation).

        Args:
            state: the state where the agent is;
            policy_state (None): the internal state of the policy. If ``None``, the stored internal state is used and
                updated;
            **kwargs: additional per-timestep conditioning inputs assembled by the agent; policies that do not
                consume them can ignore the keyword arguments.

        Returns:
            The action sampled from the policy.

        """
        if policy_state is None:
            action, self._policy_state = self._draw_action(state, self._policy_state, **kwargs)
        else:
            action, _ = self._draw_action(state, policy_state, **kwargs)

        return action

    def draw_action_greedy(self, state, policy_state=None, **kwargs):
        """
        Return the greedy action in ``state``, used during greedy evaluation.

        When ``policy_state`` is not provided, the policy uses its internal state. When it is provided, the policy uses
        the given state and leaves the internal one untouched (functional evaluation). How the internal state evolves
        is policy dependent.

        Args:
            state: the state where the agent is;
            policy_state (None): the internal state of the policy. If ``None``, the stored internal state is used;
            **kwargs: additional per-timestep conditioning inputs assembled by the agent.

        Returns:
            The greedy action of the policy.

        """
        if policy_state is None:
            action, self._policy_state = self._draw_action_greedy(state, self._policy_state, **kwargs)
        else:
            action, _ = self._draw_action_greedy(state, policy_state, **kwargs)

        return action

    def reset(self):
        """
        Reset the internal state of the policy at the beginning of an episode. Implementations must set
        ``self._policy_state`` and return it.

        Returns:
            The initial policy state.

        """
        raise NotImplementedError

    def reset_vectorized(self, start_mask):
        """
        Reset the internal state of the policy for the environments selected by ``start_mask``, leaving the other
        environments untouched. Implementations must set ``self._policy_state`` and return it.

        Args:
            start_mask: boolean mask selecting the environments that are starting a new episode.

        Returns:
            The batch of policy states after the masked reset.

        """
        raise NotImplementedError

    def stop(self):
        """
        Clear the internal state at the end of a run, so that the next run reinitializes it from scratch (and can
        therefore run with a different number of environments or a different vectorization mode).

        """
        self._policy_state = None

    def _draw_action(self, state, policy_state, **kwargs):
        """
        Sample an action in ``state`` given the policy state, returning the next policy state. This is the functional
        core of :meth:`draw_action` and must not mutate the internal state.

        Args:
            state: the state where the agent is;
            policy_state: the internal state of the policy;
            **kwargs: additional per-timestep conditioning inputs.

        Returns:
            A tuple containing the sampled action and the next policy state.

        """
        raise NotImplementedError

    def _draw_action_greedy(self, state, policy_state, **kwargs):
        """
        Return the greedy action in ``state`` given the policy state, returning the next policy state. This is
        the functional core of :meth:`draw_action_greedy` and must not mutate the internal state. Policies that
        do not define a greedy action must not override this method.

        Args:
            state: the state where the agent is;
            policy_state: the internal state of the policy;
            **kwargs: additional per-timestep conditioning inputs.

        Returns:
            A tuple containing the greedy action and the next policy state.

        """
        raise NotImplementedError("'{}' does not define a greedy action.".format(type(self).__name__))


class PolicyMixin:
    """
    Base class of the policy mixins. A class combining mixins with a :class:`Policy` subclass must list every mixin
    before the policy class in its base list; a class combining only mixins, which is not a policy itself, must be
    declared with ``is_mixin=True``.

    """
    def __init_subclass__(cls, is_mixin=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_mixin:
            mixins = [base for base in cls.__mro__[1:] if issubclass(base, PolicyMixin) and base is not PolicyMixin]

            if not issubclass(cls, Policy):
                raise TypeError(
                    "'{}' uses the {} mixin but does not inherit from Policy (intermediate mixins must pass "
                    "is_mixin=True).".format(cls.__name__, mixins[0].__name__)
                )

            for mixin in mixins:
                if cls.__mro__.index(Policy) < cls.__mro__.index(mixin):
                    raise TypeError(
                        "'{}' resolves Policy before the {} mixin. Mixins must be listed first, e.g. "
                        "'class {}({}, Policy)'.".format(cls.__name__, mixin.__name__, cls.__name__, mixin.__name__)
                    )


class HasWeights(PolicyMixin, is_mixin=True):
    """
    Mixin adding a set of trainable parameters (the policy weights) to a policy. It must be combined with a
    :class:`Policy` subclass and placed before it in the base list (e.g. ``class MyPolicy(HasWeights, Policy)`` or
    ``class MyPolicy(HasWeights, StatefulPolicy)``); on its own it is not a policy. If the policy is also
    differentiable, use :class:`HasGradient` instead.

    """
    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        raise NotImplementedError

    @property
    def weights_size(self):
        """
        Property.

        Returns:
             The size of the policy weights.

        """
        raise NotImplementedError


class HasLocation(PolicyMixin, is_mixin=True):
    """
    Mixin adding the location of the action distribution to a policy. It must be combined with a :class:`Policy`
    subclass and placed before it in the base list (e.g. ``class MyPolicy(HasLocation, Policy)``); on its own it is
    not a policy.

    """
    def location(self, state, **kwargs):
        """
        Compute the location of the action distribution in ``state``, i.e. the action of the policy without
        exploration noise. For a torch policy, the result is differentiable with respect to the policy parameters.

        Args:
            state: the state where the agent is;
            **kwargs: additional per-timestep conditioning inputs assembled by the agent.

        Returns:
            The location of the action distribution.

        """
        raise NotImplementedError


class HasGradient(HasWeights, is_mixin=True):
    """
    Mixin for a parametric policy that is also differentiable.

    It applies to the policies for which the gradient of the log-probability w.r.t. the policy weights can be
    computed, and extends :class:`HasWeights` with that derivative; policies that carry weights but are not
    differentiable should use :class:`HasWeights` directly.

    """
    def diff_log(self, state, action, policy_state=None):
        """
        Compute the gradient of the logarithm of the probability density
        function, in the specified state and action pair, i.e.:

        .. math::
            \\nabla_{\\theta}\\log p(s,a)


        Args:
            state: the state where the gradient is computed;
            action: the action where the gradient is computed;
            policy_state: the internal state of the policy.

        Returns:
            The gradient of the logarithm of the pdf w.r.t. the policy weights
        """
        raise NotImplementedError

    def diff(self, state, action, policy_state=None):
        """
        Compute the derivative of the probability density function, in the
        specified state and action pair. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick, i.e.:

        .. math::
            \\nabla_{\\theta}p(s,a)=p(s,a)\\nabla_{\\theta}\\log p(s,a)


        Args:
            state: the state where the derivative is computed;
            action: the action where the derivative is computed;
            policy_state: the internal state of the policy.

        Returns:
            The derivative w.r.t. the  policy weights
        """
        return self(state, action) * self.diff_log(state, action, policy_state)


class HasMeanApproximator(HasWeights, HasLocation, is_mixin=True):
    """
    Mixin implementing a policy built around a mean approximator: the weights and the trainable parameters of the
    policy are those of the approximator, and the location of the action distribution is its output. It must be
    combined with a :class:`Policy` subclass and placed before it in the base list.

    """
    def __init__(self, mu, **kwargs):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator computing the mean action in each state;
            **kwargs: the remaining keyword arguments, passed to the constructor of the base policy.

        """
        super().__init__(**kwargs)

        self._approximator = mu
        self._predict_params = dict()

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle'
        )

    def location(self, state):
        """
        Compute the location of the action distribution in ``state``, i.e. the output of the mean approximator,
        without exploration noise or clipping. For a torch approximator, the result is differentiable with respect to
        the policy parameters.

        Args:
            state: the state where the agent is.

        Returns:
            The location of the action distribution.

        """
        return self._approximator(state, **self._predict_params)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    def parameters(self):
        """
        Returns:
            The trainable parameters of the mean approximator.

        """
        return self._approximator.parameters()

    @property
    def weights_size(self):
        return self._approximator.weights_size
