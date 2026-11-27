import numpy as np

from mushroom_rl.policy.policy import Policy, HasGradient, HasLocation
from mushroom_rl.rl_utils.parameters import CovarianceParameter
from scipy.stats import multivariate_normal


class AbstractGaussianPolicy(HasGradient, HasLocation, Policy):
    """
    Abstract class of Gaussian policies.

    """
    def __init__(self, mu):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the mean w.r.t. the state.

        """
        super().__init__('numpy')

        self._mu_approximator = mu
        self._predict_params = dict()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _predict_params='pickle'
        )

    def draw_action_greedy(self, state):
        return self.location(state)

    def location(self, state):
        return self._mu_approximator.predict(state, **self._predict_params)


class GaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy.
    This is a differentiable policy for continuous action spaces.
    The policy samples an action in every state following a gaussian
    distribution, where the mean is computed in the state and the covariance
    matrix is a :class:`CovarianceParameter`, updated at every drawn action.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the mean w.r.t. the
                state;
            sigma ([Array, Parameter, CovarianceParameter]): the covariance of the policy: a square positive
                definite covariance matrix of size n x n, a vector of size n with the standard deviation of each action
                component, a :class:`Parameter` returning such a vector, or a :class:`CovarianceParameter`; n is the
                action dimensionality.

        """
        super().__init__(mu)

        self._sigma = CovarianceParameter.make(sigma, backend='numpy')

        self._add_save_attr(_sigma='mushroom')
        self._add_logger_attr(_sigma='sigma', group='policy')

    def __call__(self, state, action):
        return multivariate_normal.pdf(action, self.location(state), self._sigma.get_value())

    def draw_action(self, state):
        self._sigma.update()
        mu = self.location(state)

        return mu + np.random.standard_normal(mu.shape) @ self._sigma.get_cholesky().T

    def set_sigma(self, sigma):
        """
        Setter.

        Args:
            sigma ([Array, Parameter, CovarianceParameter]): the new covariance of the policy, in any of the
                forms accepted by the constructor.

        """
        self._sigma = CovarianceParameter.make(sigma, backend='numpy')
        self.set_logger(self._logger, self._log_prefix, self._log_label)

    def diff_log(self, state, action, policy_state=None):
        mu = self.location(state)
        inv_sigma = self._sigma.get_inverse()

        delta = action - mu

        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        return g

    def set_weights(self, weights):
        self._mu_approximator.set_weights(weights)

    def get_weights(self):
        return self._mu_approximator.get_weights()

    @property
    def weights_size(self):
        return self._mu_approximator.weights_size


class AbstractDiagonalGaussianPolicy(AbstractGaussianPolicy):
    """
    Abstract class of Gaussian policies with a diagonal covariance matrix.

    """
    def __call__(self, state, action):
        variance = self._variance(state)

        return np.exp(-.5 * np.sum(np.log(2 * np.pi * variance) + (action - self.location(state))**2 / variance,
                                   axis=-1))

    def draw_action(self, state):
        mu = self.location(state)

        return mu + np.sqrt(self._variance(state)) * np.random.standard_normal(mu.shape)

    def _variance(self, state):
        """
        Compute the variance of each action component.

        Args:
            state: the state, or batch of states, where the variance is computed.

        Returns:
            The variance, broadcastable against the action drawn in ``state``.

        """
        raise NotImplementedError


class DiagonalGaussianPolicy(AbstractDiagonalGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation. The Covariance matrix is constrained to be a diagonal matrix,
    where the diagonal is the squared standard deviation vector. This is a differentiable policy for continuous action
    spaces. This policy is similar to the gaussian policy, but the weights includes also the standard deviation.

    """
    def __init__(self, mu, std):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the mean w.r.t. the
                state;
            std (np.ndarray): a vector of standard deviations. The length of
                this vector must be equal to the action dimensionality.

        """
        super().__init__(mu)

        self._std = std.copy()

        self._add_save_attr(_std='numpy')

    def set_std(self, std):
        """
        Setter.

        Args:
            std (np.ndarray): the new vector of standard deviations. The length of this vector must be equal to the
                action dimensionality.

        """
        self._std[:] = std

    def diff_log(self, state, action, policy_state=None):
        mu = self.location(state)
        inv_sigma = np.diag(1. / self._std**2)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g_mu = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        # Compute standard deviation derivative
        g_sigma = -1. / self._std + delta**2 / self._std**3

        return np.concatenate((g_mu, g_sigma), axis=0)

    def set_weights(self, weights):
        self._mu_approximator.set_weights(
            weights[0:self._mu_approximator.weights_size])
        self._std[:] = weights[self._mu_approximator.weights_size:]

    def get_weights(self):
        return np.concatenate((self._mu_approximator.get_weights(), self._std),
                              axis=0)

    @property
    def weights_size(self):
        return self._mu_approximator.weights_size + self._std.size

    def _variance(self, state):
        return self._std**2


class StateStdGaussianPolicy(AbstractDiagonalGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation.
    The Covariance matrix is
    constrained to be a diagonal matrix, where the diagonal is the squared
    standard deviation, which is computed for each state.
    This is a differentiable policy for continuous action spaces.
    This policy is similar to the diagonal gaussian policy, but a parametric
    approximator is used to compute the standard deviation, so the standard
    deviation depends on the current state.

    """
    def __init__(self, mu, std, eps=1e-6):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the mean w.r.t. the
                state;
            std (Approximator): the approximator representing the standard
                deviations w.r.t. the state. The output dimensionality of the
                approximator must be equal to the action dimensionality;
            eps(float, 1e-6): A positive constant added to the variance to
                ensure that is always greater than zero.

        """
        super().__init__(mu)

        assert eps > 0

        self._std_approximator = std
        self._eps = eps

        self._add_save_attr(
            _std_approximator='mushroom',
            _eps='primitive'
        )

    def diff_log(self, state, action, policy_state=None):
        mu = self.location(state)
        std = self._std_approximator.predict(state, **self._predict_params)
        variance = std**2 + self._eps

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        sigma_inv = np.diag(1 / variance)

        g_mu = j_mu.dot(sigma_inv).dot(delta.T)

        # Compute variance derivative
        w = (delta**2 - variance) * std / variance**2
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

    def _variance(self, state):
        return self._std_approximator.predict(state, **self._predict_params)**2 + self._eps


class StateLogStdGaussianPolicy(AbstractDiagonalGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation.
    The Covariance matrix is
    constrained to be a diagonal matrix, the diagonal is computed by an
    exponential transformation of the logarithm of the standard deviation
    computed in each state.
    This is a differentiable policy for continuous action spaces.
    This policy is similar to the State std gaussian policy, but here the
    approximator represents the logarithm of the standard deviation.

    """
    def __init__(self, mu, log_std):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the mean w.r.t. the
                state;
            log_std (Approximator): an approximator representing the logarithm of the
                standard deviation w.r.t. the state. The output dimensionality of the
                approximator must be equal to the action dimensionality.

        """
        super().__init__(mu)

        self._log_std_approximator = log_std

        self._add_save_attr(_log_std_approximator='mushroom')

    def diff_log(self, state, action, policy_state=None):
        mu = self.location(state)
        variance = self._variance(state)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._mu_approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        sigma_inv = np.diag(1 / variance)

        g_mu = j_mu.dot(sigma_inv).dot(delta.T)

        # Compute variance derivative
        w = delta**2 / variance
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

    def _variance(self, state):
        return np.exp(self._log_std_approximator.predict(state, **self._predict_params))**2
