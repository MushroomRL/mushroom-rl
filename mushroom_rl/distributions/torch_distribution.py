import torch

from mushroom_rl.distributions import Distribution


class AbstractGaussianTorchDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.

    """

    def __init__(self):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            std (np.ndarray): initial vector of standard deviations for each
                variable of the distribution.

        """
        super().__init__()

    def distribution(self, initial_state=None, **context):
        mu, chol_sigma = self._get_mean_and_chol(initial_state, **context)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def sample(self, initial_state=None, **context):
        dist = self.distribution(initial_state, **context)

        if initial_state is None:
            return dist.sample()
        else:
            return dist.sample(initial_state.shape)

    def log_pdf(self, theta, initial_state=None, **context):
        dist = self.distribution(initial_state, **context)
        return dist.log_prob(theta)

    def __call__(self, theta, initial_state=None, **context):
        return torch.exp(self.log_pdf(theta, initial_state, **context))

    def entropy(self, initial_state=None, **context):
        dist = self.distribution(initial_state, **context)
        return dist.entropy()

    def mle(self, theta, weights=None):
        raise NotImplementedError

    def con_wmle(self, theta, weights, eps, kappa):
        raise NotImplementedError

    def diff_log(self, theta, initial_state=None, **context):
        raise NotImplementedError

    def _get_mean_and_chol(self, initial_state, **context):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class DiagonalGaussianTorchDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu, sigma):
        self._mu = torch.nn.Parameter(mu)
        self._log_sigma = torch.nn.Parameter(torch.log(sigma))

        super().__init__()

        self._add_save_attr(
            _mu='torch',
            _log_sigma='torch'
        )

    def get_parameters(self):
        rho = torch.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._log_sigma

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu.data = rho[:n_dims]
        self._log_sigma.data = rho[n_dims:]

    @property
    def parameters_size(self):
        return 2 * len(self._mu)

    def parameters(self):
        return [self._mu, self._log_sigma]

    def _get_mean_and_chol(self, initial_state, **context):
        return self._mu, torch.diag(torch.exp(self._log_sigma))
