from mushroom_rl.core import Serializable


class Distribution(Serializable):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization algorithms in order to perform exploration in
    parameter space. In the literature, they are also known as high level policies.

    """

    def sample(self, initial_state=None, **context):
        """
        Draw a sample from the distribution.

        Args:
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            A random vector sampled from the distribution.

        """
        raise NotImplementedError

    def log_pdf(self, theta, initial_state=None, **context):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            theta (np.ndarray): the point where the log pdf is calculated;
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def __call__(self, theta, initial_state=None, **context):
        """
        Compute the probability density function in the specified point

        Args:
            theta (np.ndarray): the point where the pdf is calculated;
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            The value of the pdf in the specified point.

        """
        raise NotImplementedError

    def entropy(self, initial_state=None, **context):
        """
        Compute the entropy of the distribution.

        Args:
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            The value of the entropy of the distribution.

        """
        raise NotImplementedError

    def mle(self, theta, weights=None):
        """
        Compute the (weighted) maximum likelihood estimate of the points,
        and update the distribution accordingly.

        Args:
            theta (np.ndarray): a set of points, every row is a sample;
            weights (np.ndarray, None): a vector of weights. If specified the weighted maximum likelihood estimate is
                computed instead of the plain maximum likelihood. The number of elements of this vector must be equal to
                the number of rows of the theta matrix.

        """
        raise NotImplementedError

    def diff_log(self, theta, initial_state=None, **context):
        """
        Compute the derivative of the logarithm of the probability density function in the specified point.

        Args:
            theta (np.ndarray): the point where the gradient of the log pdf is computed;
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            The gradient of the log pdf in the specified point.

        """
        raise NotImplementedError

    def diff(self, theta, initial_state=None, **context):
        """
        Compute the derivative of the probability density function, in the specified point. Normally it is computed
        w.r.t. the derivative of the logarithm of the probability density function, exploiting the likelihood ratio
        trick, i.e.:

        .. math::
            \\nabla_{\\rho}p(\\theta)=p(\\theta)\\nabla_{\\rho}\\log p(\\theta)

        Args:
            theta (np.ndarray): the point where the gradient of the pdf is calculated;
            initial_state (array, None): First state encountered in the environment;
            **context: context variables to condition the distribution.

        Returns:
            The gradient of the pdf in the specified point.

        """
        return self(theta, initial_state, **context) * self.diff_log(theta, initial_state, **context)

    def get_parameters(self):
        """
        Getter.

        Returns:
             The current distribution parameters.

        """
        raise NotImplementedError

    def set_parameters(self, rho):
        """
        Setter.

        Args:
            rho (np.ndarray): the vector of the new parameters to be used by the distribution.

        """
        raise NotImplementedError

    @property
    def parameters_size(self):
        """
        Property.

        Returns:
             The size of the distribution parameters.

        """
        raise NotImplementedError
