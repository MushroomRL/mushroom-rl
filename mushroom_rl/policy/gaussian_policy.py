import numpy as np

from .policy import ParametricPolicy
from scipy.stats import multivariate_normal


class AbstractGaussianPolicy(ParametricPolicy):
    """
    Abstract class of Gaussian policies.

    """
    def __init__(self, policy_state_shape=None):
        """
        Constructor.

        """
        super().__init__(policy_state_shape)

    def __call__(self, state, action, policy_state=None):
        mu, sigma = self._compute_multivariate_gaussian(state)[:2]

        return multivariate_normal.pdf(action, mu, sigma)

    def draw_action(self, state, policy_state=None):
        mu, sigma = self._compute_multivariate_gaussian(state)[:2]

        return np.random.multivariate_normal(mu, sigma), None


class GaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy.
    This is a differentiable policy for continuous action spaces.
    The policy samples an action in every state following a gaussian
    distribution, where the mean is computed in the state and the covariance
    matrix is fixed.

    """
    def __init__(self, mu, sigma, policy_state_shape=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            sigma (np.ndarray): a square positive definite matrix representing
                the covariance matrix. The size of this matrix must be n x n,
                where n is the action dimensionality.

        """
        super().__init__(policy_state_shape)

        self._approximator = mu
        self._predict_params = dict()
        self._inv_sigma = np.linalg.inv(sigma)
        self._sigma = sigma

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _inv_sigma='numpy',
            _sigma='numpy'
        )

    def set_sigma(self, sigma):
        """
        Setter.

        Args:
            sigma (np.ndarray): the new covariance matrix. Must be a square
                positive definite matrix.

        """
        self._sigma = sigma
        self._inv_sigma = np.linalg.inv(sigma)

    def diff_log(self, state, action, policy_state=None):
        mu, _, inv_sigma = self._compute_multivariate_gaussian(state)

        delta = action - mu

        j_mu = self._approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        return g

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)

        return mu, self._sigma, self._inv_sigma


class DiagonalGaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation. The Covariance matrix is constrained to be a diagonal matrix,
    where the diagonal is the squared standard deviation vector. This is a differentiable policy for continuous action
    spaces. This policy is similar to the gaussian policy, but the weights includes also the standard deviation.

    """
    def __init__(self, mu, std, policy_state_shape=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            std (np.ndarray): a vector of standard deviations. The length of
                this vector must be equal to the action dimensionality.

        """
        super().__init__(policy_state_shape)

        self._approximator = mu
        self._predict_params = dict()
        self._std = std

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _std='numpy'
        )

    def set_std(self, std):
        """
        Setter.

        Args:
            std (np.ndarray): the new standard deviation. Must be a square
                positive definite matrix.

        """
        self._std = std

    def diff_log(self, state, action, policy_state=None):
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

    def set_weights(self, weights):
        self._approximator.set_weights(
            weights[0:self._approximator.weights_size])
        self._std = weights[self._approximator.weights_size:]

    def get_weights(self):
        return np.concatenate((self._approximator.get_weights(), self._std),
                              axis=0)

    @property
    def weights_size(self):
        return self._approximator.weights_size + self._std.size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)

        sigma = self._std**2

        return mu, np.diag(sigma), np.diag(1. / sigma)


class StateStdGaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation.
    The Covariance matrix is
    constrained to be a diagonal matrix, where the diagonal is the squared
    standard deviation, which is computed for each state.
    This is a differentiable policy for continuous action spaces.
    This policy is similar to the diagonal gaussian policy, but a parametric
    regressor is used to compute the standard deviation, so the standard
    deviation depends on the current state.

    """
    def __init__(self, mu, std, eps=1e-6, policy_state_shape=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            std (Regressor): the regressor representing the standard
                deviations w.r.t. the state. The output dimensionality of the
                regressor must be equal to the action dimensionality;
            eps(float, 1e-6): A positive constant added to the variance to
                ensure that is always greater than zero.

        """
        assert(eps > 0)

        super().__init__(policy_state_shape)

        self._mu_approximator = mu
        self._std_approximator = std
        self._predict_params = dict()
        self._eps = eps

        self._add_save_attr(
            _mu_approximator='mushroom',
            _std_approximator='mushroom',
            _predict_params='pickle',
            _eps='primitive'
        )

    def diff_log(self, state, action, policy_state=None):
        mu, sigma, std = self._compute_multivariate_gaussian(state)
        diag_sigma = np.diag(sigma)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        sigma_inv = np.diag(1 / diag_sigma)

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
            np.expand_dims(state, axis=0), **self._predict_params), -1)

        std = np.reshape(self._std_approximator.predict(
            np.expand_dims(state, axis=0), **self._predict_params), -1)

        sigma = std**2 + self._eps

        return mu, np.diag(sigma), std


class StateLogStdGaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation.
    The Covariance matrix is
    constrained to be a diagonal matrix, the diagonal is computed by an
    exponential transformation of the logarithm of the standard deviation
    computed in each state.
    This is a differentiable policy for continuous action spaces.
    This policy is similar to the State std gaussian policy, but here the
    regressor represents the logarithm of the standard deviation.

    """
    def __init__(self, mu, log_std, policy_state_shape=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            log_std (Regressor): a regressor representing the logarithm of the
                variance w.r.t. the state. The output dimensionality of the
                regressor must be equal to the action dimensionality.

        """
        super().__init__(policy_state_shape)

        self._mu_approximator = mu
        self._log_std_approximator = log_std
        self._predict_params = dict()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _log_std_approximator='mushroom',
            _predict_params='pickle'
        )

    def diff_log(self, state, action, policy_state=None):

        mu, sigma = self._compute_multivariate_gaussian(state)
        diag_sigma = np.diag(sigma)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        sigma_inv = np.diag(1 / diag_sigma)

        g_mu = j_mu.dot(sigma_inv).dot(delta.T)

        # Compute variance derivative
        w = delta**2 / diag_sigma
        j_sigma = np.atleast_2d(self._log_std_approximator.diff(state).T)
        g_sigma = np.atleast_1d(w.dot(j_sigma)) - np.sum(j_sigma, axis=0)

        return np.concatenate((g_mu, g_sigma), axis=0)

    def set_weights(self, weights):
        mu_weights = weights[0:self._mu_approximator.weights_size]
        log_std_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._log_std_approximator.set_weights(log_std_weights)

    def get_weights(self):
        mu_weights = self._mu_approximator.get_weights()
        log_std_weights = self._log_std_approximator.get_weights()

        return np.concatenate((mu_weights, log_std_weights), axis=0)

    @property
    def weights_size(self):
        return self._mu_approximator.weights_size + self._log_std_approximator.weights_size

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._mu_approximator.predict(
            np.expand_dims(state, axis=0), **self._predict_params), -1)

        log_std = np.reshape(self._log_std_approximator.predict(
            np.expand_dims(state, axis=0), **self._predict_params), -1)

        sigma = np.exp(log_std)**2

        return mu, np.diag(sigma)
