import numpy as np

from mushroom.utils.parameters import Parameter
from scipy.stats import norm, multivariate_normal


class GaussianPolicy:
    def __init__(self, mu, sigma):
        self.__name__ = 'GaussianPolicy'

        assert isinstance(sigma, Parameter)

        self._approximator = mu
        self._sigma = sigma

    def __call__(self, state, action):
        mu, sigma = self._compute_gaussian(state, False)

        return norm.pdf(action[0], mu[0], sigma)

    def draw_action(self, state):
        mu, sigma = self._compute_gaussian(state)

        return np.random.normal(mu, sigma)

    def diff(self, state, action):
        return self(state, action) * self.diff_log(state, action)

    def diff_log(self, state, action):
        mu, sigma = self._compute_gaussian(state, False)
        delta = action - mu
        g_mu = np.expand_dims(self._approximator.diff(state), axis=1)

        g = g_mu.dot(delta) / sigma**2

        return g

    def set_sigma(self, sigma):
        assert isinstance(sigma, Parameter)

        self._sigma = sigma

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def _compute_gaussian(self, state, update=True):
        if update:
            sigma = self._sigma(state)
        else:
            sigma = self._sigma.get_value(state)
        mu = np.reshape(self._approximator.predict(np.expand_dims(state,
                                                                  axis=0)), -1)

        return mu, sigma

    def __str__(self):
        return self.__name__


class MultivariateGaussianPolicy:
    def __init__(self, mu, sigma):
        self.__name__ = 'MultivariateGaussianPolicy'

        self._approximator = mu
        self._inv_sigma = np.linalg.inv(sigma)
        self._sigma = sigma

    def __call__(self, state, action):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return multivariate_normal.pdf(action, mu, sigma)

    def draw_action(self, state):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return np.random.multivariate_normal(mu, sigma)

    def diff(self, state, action):
        return self(state, action) * self.diff_log(state, action)

    def diff_log(self, state, action):

        mu, _, inv_sigma = self._compute_multivariate_gaussian(state)

        delta = action - mu

        g_mu = self._approximator.diff(state)

        if len(g_mu.shape) == 1:
            g_mu = np.expand_dims(g_mu, axis=1)

        g = .5 * g_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        return g

    def set_sigma(self, sigma):
        self._sigma = sigma
        self._inv_sigma = np.linalg.inv(sigma)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state,
                                                                  axis=0)), -1)

        return mu, self._sigma, self._inv_sigma

    def __str__(self):
        return self.__name__


class MultivariateDiagonalGaussianPolicy:
    def __init__(self, mu, sigma):
        self.__name__ = 'MultivariateDiagonalGaussianPolicy'

        self._approximator = mu
        self._sigma = sigma

    def __call__(self, state, action):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return multivariate_normal.pdf(action, mu, sigma)

    def draw_action(self, state):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return np.random.multivariate_normal(mu, sigma)

    def diff(self, state, action):
        return self(state, action) * self.diff_log(state, action)

    def diff_log(self, state, action):

        mu, _, inv_sigma = self._compute_multivariate_gaussian(state)

        delta = action - mu

        g_mu = self._approximator.diff(state)

        if len(g_mu.shape) == 1:
            g_mu = np.expand_dims(g_mu, axis=1)

        g = .5 * g_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        g_sigma = -1. / self._sigma + delta**2 / self._sigma**3

        return np.concatenate((g, g_sigma), axis=0)

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_weights(self, weights):
        self._approximator.set_weights(
            weights[0:self._approximator.weights_size])
        self._sigma = weights[self._approximator.weights_size:]

    def get_weights(self):
        return np.concatenate((self._approximator.get_weights(), self._sigma), axis=0)

    @property
    def weights_size(self):
        return self._approximator.weights_size + self._sigma.size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state,
                                                                  axis=0)), -1)

        sigma2 = self._sigma**2

        return mu, np.diag(sigma2), np.diag(1. / sigma2)

    def __str__(self):
        return self.__name__
