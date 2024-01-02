from mushroom_rl.core import Serializable


class Distribution(Serializable):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization algorithms in order to perform exploration in
    parameter space. In the literature, they are also known as high level policies.

    """
    def __init__(self, context_shape=None):
        self._context_shape = context_shape

        super().__init__()

        self._add_save_attr(_context_shape='primitive')

    def sample(self, context=None):
        """
        Draw a sample from the distribution.

        Args:
            context (Array, None): context variables to condition the distribution.

        Returns:
            A random vector sampled from the distribution.

        """
        raise NotImplementedError

    def log_pdf(self, theta, context=None):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            theta (np.ndarray): the point where the log pdf is calculated;
            context (Array, None): context variables to condition the distribution.

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def __call__(self, theta, context=None):
        """
        Compute the probability density function in the specified point

        Args:
            theta (np.ndarray): the point where the pdf is calculated;
            context (Array, None): context variables to condition the distribution.

        Returns:
            The value of the pdf in the specified point.

        """
        raise NotImplementedError

    def mean(self, context=None):
        raise NotImplementedError

    def entropy(self, context=None):
        """
        Compute the entropy of the distribution.

        Args:
            context (Array, None): context variables to condition the distribution.

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

    def diff_log(self, theta, context=None):
        """
        Compute the derivative of the logarithm of the probability density function in the specified point.

        Args:
            theta (np.ndarray): the point where the gradient of the log pdf is computed;
            context (Array, None): context variables to condition the distribution.

        Returns:
            The gradient of the log pdf in the specified point.

        """
        raise NotImplementedError

    def diff(self, theta, context=None):
        """
        Compute the derivative of the probability density function, in the specified point. Normally it is computed
        w.r.t. the derivative of the logarithm of the probability density function, exploiting the likelihood ratio
        trick, i.e.:

        .. math::
            \\nabla_{\\rho}p(\\theta)=p(\\theta)\\nabla_{\\rho}\\log p(\\theta)

        Args:
            theta (np.ndarray): the point where the gradient of the pdf is calculated;
            context (Array, None): context variables to condition the distribution.

        Returns:
            The gradient of the pdf in the specified point.

        """
        return self(theta, context) * self.diff_log(theta, context)

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

    @property
    def is_contextual(self):
        return self._context_shape is not None
