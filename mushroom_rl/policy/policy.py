from mushroom_rl.core import Serializable


class Policy(Serializable):
    """
    Interface representing a generic policy.
    A policy is a probability distribution that gives the probability of taking
    an action given a specified state.
    A policy is used by mushroom agents to interact with the environment.

    """
    def __init__(self, policy_state_shape=None):
        """
        Constructor.

        Args:
            policy_state_shape (tuple, None): the shape of the internal state of the policy.

        """
        super().__init__()

        self.policy_state_shape = policy_state_shape

        self._add_save_attr(policy_state_shape='primitive')

    def __call__(self, state, action, policy_state):
        """
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            state: state where you want to evaluate the policy density;
            action: action where you want to evaluate the policy density;
            policy_state: internal_state where you want to evaluate the policy density.

        Returns:
            The probability of all actions following the policy in the given state if the list contains only the state,
            else the probability of the given action in the given state following the policy. If the action space is
            continuous, state and action must be provided

        """
        raise NotImplementedError

    def draw_action(self, state, policy_state):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state: the state where the agent is;
            policy_state: the internal state of the policy.

        Returns:
            The action sampled from the policy and optionally the next policy state.

        """
        raise NotImplementedError

    def reset(self):
        """
        Useful when the policy needs a special initialization at the beginning
        of an episode.

        Returns:
            The initial policy state (by default None).

        """
        return None

    @property
    def is_stateful(self):
        return self.policy_state_shape is not None


class ParametricPolicy(Policy):
    """
    Interface for a generic parametric policy.
    A parametric policy is a policy that depends on set of parameters, called the policy weights.
    For differentiable policies, the derivative of the probability for a specified state-action pair can be provided.

    """

    def __init__(self, policy_state_shape=None):
        """
        Constructor.

        Args:
            policy_state_shape (tuple, None): the shape of the internal state of the policy.

        """
        super().__init__(policy_state_shape)

    def diff_log(self, state, action, policy_state):
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
        raise RuntimeError('The policy is not differentiable')

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
        return self(state, action, policy_state) * self.diff_log(state, action, policy_state)

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
