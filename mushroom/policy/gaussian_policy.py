import numpy as np

from mushroom.utils.parameters import Parameter
from scipy.stats import norm, multivariate_normal


class GaussianPolicy:
    def __init__(self, mu, sigma):
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
        j_mu = np.expand_dims(self._approximator.diff(state), axis=1)

        g = j_mu.dot(delta) / sigma**2

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


class MultivariateGaussianPolicy:
    def __init__(self, mu, sigma):
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

        j_mu = self._approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

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


class MultivariateDiagonalGaussianPolicy:
    def __init__(self, mu, std):
        self._approximator = mu
        self._std = std

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

        # Compute mean derivative
        j_mu = self._approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g_mu = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        # Compute standard deviation derivative
        g_sigma = -1. / self._std + delta**2 / self._std**3

        return np.concatenate((g_mu, g_sigma), axis=0)

    def set_std(self, std):
        self._std = std

    def set_weights(self, weights):
        self._approximator.set_weights(
            weights[0:self._approximator.weights_size])
        self._std = weights[self._approximator.weights_size:]

    def get_weights(self):
        return np.concatenate((self._approximator.get_weights(), self._std), axis=0)

    @property
    def weights_size(self):
        return self._approximator.weights_size + self._std.size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state,
                                                                  axis=0)), -1)

        sigma = self._std**2

        return mu, np.diag(sigma), np.diag(1. / sigma)


class MultivariateStateStdGaussianPolicy:
    def __init__(self, mu, std, eps=1e-6):
        assert(eps > 0)

        self._mu_approximator = mu
        self._std_approximator = std
        self._eps = eps

    def __call__(self, state, action):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return multivariate_normal.pdf(action, mu, sigma)

    def draw_action(self, state):
        mu, sigma, _ = self._compute_multivariate_gaussian(state)

        return np.random.multivariate_normal(mu, sigma)

    def diff(self, state, action):
        return self(state, action) * self.diff_log(state, action)

    def diff_log(self, state, action):

        mu, sigma, std = self._compute_multivariate_gaussian(state)
        diag_sigma = np.diag(sigma)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        sigma_inv = np.diag(1/diag_sigma)

        g_mu = j_mu.dot(sigma_inv).dot(delta.T)

        # Compute variance derivative
        w = (delta**2 - diag_sigma) * std / diag_sigma**2
        j_sigma = np.atleast_2d(self._std_approximator.diff(state).T)
        g_sigma = np.atleast_1d(w.dot(j_sigma))

        return np.concatenate((g_mu, g_sigma), axis=0)

    def set_weights(self, weights):
        mu_weights = weights[0:self._mu_approximator.weights_size]
        std_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._std_approximator.set_weights(std_weights)

    def get_weights(self):
        mu_weights = self._mu_approximator.get_weights()
        std_weights = self._std_approximator.get_weights()

        return np.concatenate((mu_weights, std_weights), axis=0)

    @property
    def weights_size(self):
        return self._mu_approximator.weights_size + \
               self._std_approximator.weights_size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._mu_approximator.predict(
            np.expand_dims(state, axis=0)), -1)

        std = np.reshape(self._std_approximator.predict(
            np.expand_dims(state, axis=0)), -1)

        sigma = std**2 + self._eps

        return mu, np.diag(sigma), std
