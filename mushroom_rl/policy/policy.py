from mushroom_rl.core import Serializable


class Policy(Serializable):
    """
    Interface representing a generic policy.
    A policy is a probability distribution that gives the probability of taking
    an action given a specified state.
    A policy is used by mushroom agents to interact with the environment.

    """
    def __call__(self, *args):
        """
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            *args (list): list containing a state or a state and an action.

        Returns:
            The probability of all actions following the policy in the given
            state if the list contains only the state, else the probability
            of the given action in the given state following the policy. If
            the action space is continuous, state and action must be provided

        """
        raise NotImplementedError

    def draw_action(self, state):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action sampled from the policy.

        """
        raise NotImplementedError

    def reset(self):
        """
        Useful when the policy needs a special initialization at the beginning
        of an episode.

        """
        pass


class ParametricPolicy(Policy):
    """
    Interface for a generic parametric policy.
    A parametric policy is a policy that depends on set of parameters,
    called the policy weights.
    If the policy is differentiable, the derivative of the probability for a
    specified state-action pair can be provided.
    """

    def diff_log(self, state, action):
        """
        Compute the gradient of the logarithm of the probability density
        function, in the specified state and action pair, i.e.:

        .. math::
            \\nabla_{\\theta}\\log p(s,a)


        Args:
            state (np.ndarray): the state where the gradient is computed
            action (np.ndarray): the action where the gradient is computed

        Returns:
            The gradient of the logarithm of the pdf w.r.t. the policy weights
        """
        raise RuntimeError('The policy is not differentiable')

    def diff(self, state, action):
        """
        Compute the derivative of the probability density function, in the
        specified state and action pair. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick, i.e.:

        .. math::
            \\nabla_{\\theta}p(s,a)=p(s,a)\\nabla_{\\theta}\\log p(s,a)


        Args:
            state (np.ndarray): the state where the derivative is computed
            action (np.ndarray): the action where the derivative is computed

        Returns:
            The derivative w.r.t. the  policy weights
        """
        return self(state, action) * self.diff_log(state, action)

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
