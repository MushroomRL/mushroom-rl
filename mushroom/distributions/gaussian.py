import numpy as np
from scipy.stats import multivariate_normal


class GaussianDistribution:
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        inv_sigma = np.linalg.inv(sigma)
        self._K = inv_sigma + inv_sigma.T

    def sample(self):
        return np.random.multivariate_normal(self._mu, self._sigma)

    def log_pdf(self, theta):
        return multivariate_normal.logpdf(theta, self._mu, self._sigma)

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta)
        else:
            self._mu = weights.dot(theta)/np.sum(weights)

    def diff_log(self, theta):
        delta = theta - self._mu
        g = .5 * self._mu.dot(self._K).dot(delta.T)
        return g

    def get_parameters(self):
        return self._mu

    def set_parameters(self, rho):
        self._mu = rho

    @property
    def parameters_size(self):
        return len(self._mu)