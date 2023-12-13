import numpy as np
from .distribution import Distribution
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

class GaussianDistribution(Distribution):
    """
    Gaussian distribution with fixed covariance matrix. The parameters
    vector represents only the mean.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): covariance matrix of the distribution.

        """
        self._mu = mu
        self._sigma = sigma
        self._inv_sigma = np.linalg.inv(sigma)

        self._add_save_attr(
            _mu='numpy',
            _sigma='numpy',
            _inv_sigma='numpy'
        )

    def sample(self):
        return np.random.multivariate_normal(self._mu, self._sigma)

    def log_pdf(self, theta):
        return multivariate_normal.logpdf(theta, self._mu, self._sigma)

    def __call__(self, theta):
        return multivariate_normal.pdf(theta, self._mu, self._sigma)

    def entropy(self):
        n_dims = len(self._mu)
        sigma = self._sigma
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        return GaussianDistribution._entropy(logdet_sigma, n_dims)

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
        else:
            self._mu = weights.dot(theta) / np.sum(weights)

    def con_wmle(self, theta, weights, eps, *args):
        n_dims = len(self._mu)
        mu =self._mu
        sigma = self._sigma
        
        eta_start = np.array([1000])
        res = minimize(GaussianDistribution._lagrangian_eta, eta_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(weights, theta, mu, sigma, n_dims, eps),
                       method='SLSQP')
        
        eta_opt  = res.x[0]

        self._mu = GaussianDistribution._compute_mu_from_lagrangian(weights, theta, mu, eta_opt)

    def diff_log(self, theta):
        delta = theta - self._mu
        g = self._inv_sigma.dot(delta)

        return g

    def get_parameters(self):
        return self._mu

    def set_parameters(self, rho):
        self._mu = rho

    @property
    def parameters_size(self):
        return len(self._mu)

    @staticmethod
    def _compute_mu_from_lagrangian(weights, theta, mu, eta):
        
        weights_sum = np.sum(weights)

        mu_new = (weights @ theta + eta * mu) / (weights_sum + eta)
        
        return mu_new

    @staticmethod
    def _kl_constraint(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims):
        
        return 0.5*(np.trace(sigma_new_inv@sigma) - n_dims + logdet_sigma_new - logdet_sigma + (mu_new - mu).T @ sigma_new_inv @ (mu_new - mu))
    
    @staticmethod
    def _entropy(logdet_sigma, n_dims):
        c = n_dims * np.log(2*np.pi)

        return 0.5 * (logdet_sigma + c + n_dims)

    @staticmethod
    def _lagrangian_eta(lag_array, weights, theta, mu, sigma, n_dims, eps):

        eta = lag_array[0]

        mu_new = GaussianDistribution._compute_mu_from_lagrangian(weights, theta, mu, eta)
        
        sigma_inv = np.linalg.inv(sigma)

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)

        c = n_dims * np.log(2*np.pi)

        sum1 = np.sum([w_i * (-0.5 * (theta_i - mu_new)[:,np.newaxis].T @ sigma_inv @ (theta_i -  mu_new)[:,np.newaxis] - 0.5 * logdet_sigma - 0.5 * c) for w_i, theta_i in zip(weights, theta)])
        sum2 = eta * (eps - GaussianDistribution._kl_constraint(mu, mu_new, sigma, sigma, sigma_inv, sigma_inv, logdet_sigma, logdet_sigma, n_dims))
        
        return sum1 + sum2

class GaussianDiagonalDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.

    """
    def __init__(self, mu, std):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            std (np.ndarray): initial vector of standard deviations for each
                variable of the distribution.

        """
        assert(len(std.shape) == 1)
        self._mu = mu
        self._std = std

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def sample(self):
        sigma = np.diag(self._std**2)
        return np.random.multivariate_normal(self._mu, sigma)

    def log_pdf(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        n_dims = len(self._mu)
        sigma = np.diag(self._std**2)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        return GaussianDiagonalDistribution._entropy(logdet_sigma, n_dims)

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            self._std = np.std(theta, axis=0)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta2 = (theta - self._mu)**2
            self._std = np.sqrt(weights.dot(delta2) / Z)

    def con_wmle(self, theta, weights, eps, kappa):
        n_dims = len(self._mu)
        mu = self._mu
        sigma = self._std

        eta_omg_start = np.array([1000, 0])
        res = minimize(GaussianDiagonalDistribution._lagrangian_eta_omega, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa),
                       method='SLSQP')

        eta_opt, omg_opt  = res.x[0], res.x[1]

        self._mu, self._std = GaussianDiagonalDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta_opt, omg_opt)

    def diff_log(self, theta):
        n_dims = len(self._mu)

        sigma = self._std**2

        g = np.empty(self.parameters_size)

        delta = theta - self._mu

        g_mean = delta / sigma
        g_std = delta**2 / (self._std**3) - 1 / self._std

        g[:n_dims] = g_mean
        g[n_dims:] = g_std
        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._std

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = rho[:n_dims]
        self._std = rho[n_dims:]

    @property
    def parameters_size(self):
        return 2 * len(self._mu)

    @staticmethod
    def _compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta, omg):
        weights_sum = np.sum(weights)

        mu_new = (weights @ theta + eta * mu) / (weights_sum + eta)
        sigma_new = np.sqrt( ( np.sum([w_i * (theta_i-mu_new)**2 for theta_i, w_i in zip(theta, weights)], axis=0) + eta*sigma**2 + eta*(mu_new - mu)**2 ) / ( weights_sum + eta - omg ) )

        return mu_new, sigma_new

    @staticmethod
    def _kl_constraint(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims):
        return 0.5*(np.trace(sigma_new_inv@sigma) - n_dims + logdet_sigma_new - logdet_sigma + (mu_new - mu).T @ sigma_new_inv @ (mu_new - mu))
    
    @staticmethod
    def _entropy(logdet_sigma, n_dims):
        c = n_dims * np.log(2*np.pi)
        
        return 0.5 * (logdet_sigma + c + n_dims)

    @staticmethod
    def _lagrangian_eta_omega(lag_array, weights, theta, mu, sigma, n_dims, eps, kappa):

        eta, omg = lag_array[0], lag_array[1]

        mu_new, sigma_new = GaussianDiagonalDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta, omg)
        
        sigma = np.diag(sigma**2)
        sigma_new = np.diag(sigma_new**2)

        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)

        c = n_dims * np.log(2*np.pi)

        sum1 = np.sum([w_i * (-0.5*(theta_i - mu_new)[:,np.newaxis].T @ sigma_new_inv @ (theta_i -  mu_new)[:,np.newaxis] - 0.5 * logdet_sigma_new - c/2) for w_i, theta_i in zip(weights, theta)])
        
        sum2 = eta * (eps - GaussianDiagonalDistribution._kl_constraint(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims))
        
        sum3 = omg * (GaussianDiagonalDistribution._entropy(logdet_sigma_new, n_dims) - ( GaussianDiagonalDistribution._entropy(logdet_sigma, n_dims) - kappa ) )

        return sum1 + sum2 + sum3


class GaussianCholeskyDistribution(Distribution):
    """
    Gaussian distribution with full covariance matrix. The parameters
    vector represents the mean and the Cholesky decomposition of the
    covariance matrix. This parametrization enforce the covariance matrix to be
    positive definite.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): initial covariance matrix of the distribution.

        """
        self._mu = mu
        self._chol_sigma = np.linalg.cholesky(sigma)

        self._add_save_attr(
            _mu='numpy',
            _chol_sigma='numpy'
        )

    def sample(self):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return np.random.multivariate_normal(self._mu, sigma)

    def log_pdf(self, theta):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        n_dims = len(self._mu)
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        return GaussianCholeskyDistribution._entropy(logdet_sigma, n_dims)

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            sigma = np.cov(theta, rowvar=False)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta = theta - self._mu

            sigma = delta.T.dot(np.diag(weights)).dot(delta) / Z

        self._chol_sigma = np.linalg.cholesky(sigma)

    def con_wmle(self, theta, weights, eps, kappa):
        n_dims = len(self._mu)
        mu =self._mu
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        
        eta_omg_start = np.array([1000, 0])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omega, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa),
                       method='SLSQP')
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta_opt, omg_opt)

        self._mu, self._chol_sigma = mu_new, np.linalg.cholesky(sigma_new)

    def diff_log(self, theta):
        n_dims = len(self._mu)
        inv_chol = np.linalg.inv(self._chol_sigma)
        inv_sigma = inv_chol.T.dot(inv_chol)

        g = np.empty(self.parameters_size)

        delta = theta - self._mu
        g_mean = inv_sigma.dot(delta)

        delta_a = np.reshape(delta, (-1, 1))
        delta_b = np.reshape(delta, (1, -1))

        S = inv_chol.dot(delta_a).dot(delta_b).dot(inv_sigma)

        g_cov = S - np.diag(np.diag(inv_chol))

        g[:n_dims] = g_mean
        g[n_dims:] = g_cov.T[np.tril_indices(n_dims)]

        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._chol_sigma[np.tril_indices(n_dims)]

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = rho[:n_dims]
        self._chol_sigma[np.tril_indices(n_dims)] = rho[n_dims:]

    @property
    def parameters_size(self):
        n_dims = len(self._mu)

        return 2 * n_dims + (n_dims * n_dims - n_dims) // 2

    @staticmethod
    def _compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta, omg):
        
        weights_sum = np.sum(weights)

        mu_new = (weights @ theta + eta * mu) / (weights_sum + eta)
        
        sigmawa = (theta - mu_new).T @ np.diag(weights) @ (theta - mu_new)
        sigma_new = (sigmawa + eta * sigma + eta * (mu_new - mu)[:, np.newaxis] @ (mu_new - mu)[:, np.newaxis].T) / (weights_sum + eta - omg)
        
        return mu_new, sigma_new

    @staticmethod
    def _kl_constraint(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims):
        
        return 0.5*(np.trace(sigma_new_inv@sigma) - n_dims + logdet_sigma_new - logdet_sigma + (mu_new - mu).T @ sigma_new_inv @ (mu_new - mu))
    
    @staticmethod
    def _entropy(logdet_sigma, n_dims):
        c = n_dims * np.log(2*np.pi)

        return 0.5 * (logdet_sigma + c + n_dims)

    @staticmethod
    def _lagrangian_eta_omega(lag_array, weights, theta, mu, sigma, n_dims, eps, kappa):

        eta, omg = lag_array[0], lag_array[1]

        mu_new, sigma_new = GaussianCholeskyDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta, omg)
        
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)

        c = n_dims * np.log(2*np.pi)

        sum1 = np.sum([w_i * (-0.5 * (theta_i - mu_new)[:,np.newaxis].T @ sigma_new_inv @ (theta_i -  mu_new)[:,np.newaxis] - 0.5 * logdet_sigma_new - 0.5 * c) for w_i, theta_i in zip(weights, theta)])
        
        sum2 = eta * (eps - GaussianCholeskyDistribution._kl_constraint(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims))
        
        sum3 = omg * (GaussianCholeskyDistribution._entropy(logdet_sigma_new, n_dims) - ( GaussianCholeskyDistribution._entropy(logdet_sigma, n_dims) - kappa ) )

        return sum1 + sum2 + sum3
        
