import torch

from mushroom_rl.distributions import Distribution


class AbstractGaussianTorchDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.

    """

    def __init__(self, context_shape=None):
        """
        Constructor.

        Args:
            context_shape (Tuple): shape of the context variable.

        """
        super().__init__(context_shape)

    def distribution(self, context=None):
        mu, chol_sigma = self._get_mean_and_chol(context)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def sample(self, context=None):
        dist = self.distribution(context)

        return dist.sample()

    def log_pdf(self, theta, context=None):
        dist = self.distribution(context)
        return dist.log_prob(theta)

    def __call__(self, theta, context=None):
        return torch.exp(self.log_pdf(theta, context))

    def mean(self, context=None):
        mu, _ = self._get_mean_and_chol(context)
        return mu

    def entropy(self, context=None):
        dist = self.distribution(context)
        return dist.entropy()

    def mle(self, theta, weights=None):
        raise NotImplementedError

    def con_wmle(self, theta, weights, eps, kappa):
        raise NotImplementedError

    def diff_log(self, theta, context=None):
        raise NotImplementedError

    def _get_mean_and_chol(self, context):
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

    def _get_mean_and_chol(self, context):
        return self._mu, torch.diag(torch.exp(self._log_sigma))


class CholeskyGaussianTorchDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu, sigma):
        chol_sigma = torch.linalg.cholesky(sigma)

        self._mu = torch.nn.Parameter(mu)
        self._chol_sigma = torch.nn.Parameter(chol_sigma)

        super().__init__()

        self._add_save_attr(
            _mu='torch',
            _chol_sigma='torch'
        )

    def get_parameters(self):
        rho = torch.empty(self.parameters_size)
        n_dims = len(self._mu)
        tril_indices = torch.tril_indices(row=n_dims, col=n_dims)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._chol_sigma.data[tril_indices[0], tril_indices[1]]

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        tril_indices = torch.tril_indices(row=n_dims, col=n_dims)
        self._mu.data = rho[:n_dims]
        self._chol_sigma.data[tril_indices[0], tril_indices[1]] = rho[n_dims:]

    @property
    def parameters_size(self):
        n_dims = len(self._mu)
        return 2 * n_dims + (n_dims * n_dims - n_dims) // 2

    def parameters(self):
        return [self._mu, self._chol_sigma]

    def _get_mean_and_chol(self, context):
        return self._mu, self._chol_sigma.tril()
