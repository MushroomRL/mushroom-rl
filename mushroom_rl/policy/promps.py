import numpy as np
from scipy.stats import multivariate_normal

from mushroom_rl.policy.policy import StatefulPolicy, HasWeights


class ProMP(StatefulPolicy, HasWeights):
    """
    Class representing a Probabilistic Movement Primitive (ProMP). Specifically, this class represents the low-level
    gaussian time-dependant policy.

    Differently from the original implementation of ProMPs, an arbitrary regressor can be used to compute the mean from
    time features. By using a non-linear regressor, the theory behind conditioning might not hold.

    """
    def __init__(self, mu, phi, duration, sigma=None, periodic=False):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean at each time step;
            phi (Features): Basis functions used as time features;
            duration (int): duration of the movement in number of steps;
            sigma (np.ndarray; None): a square positive definite matrix representing the covariance matrix. The size of
                this matrix must be n x n, where n is the action dimensionality. If not specified, the policy returns
                the mean value;
            periodic (bool, False): whether the movement represented is periodic or not. If true, the duration parameter
                represent the duration of a period, and the phase variable increase continuously

        """
        assert sigma is None or (len(sigma.shape) == 2 and sigma.shape[0] == sigma.shape[1])

        super().__init__(policy_state_shape=(1,))

        self._approximator = mu
        self._phi = phi
        self._duration = duration
        self._chol_sigma = np.linalg.cholesky(sigma) if sigma is not None else None
        self._periodic = periodic

        self._add_save_attr(
            _approximator='mushroom',
            _phi='mushroom',
            _duration='primitive',
            _chol_sigma='numpy',
            _periodic='primitive'
        )

    def __call__(self, state, action, policy_state=None):
        if policy_state is None:
            policy_state = self._policy_state

        z = self._compute_phase(state, policy_state)
        mu = self._approximator(self._phi(z))

        if self._chol_sigma is None:
            return 1.0 if np.array_equal(mu, action) else 0.0
        else:
            return multivariate_normal.pdf(action, mu, self._chol_sigma @ self._chol_sigma.T)

    def _draw_action(self, state, policy_state):
        if self._chol_sigma is None:
            return self._draw_action_greedy(state, policy_state)

        mu, next_policy_state = self._draw_action_greedy(state, policy_state)

        return mu + np.random.randn(*mu.shape) @ self._chol_sigma.T, next_policy_state

    def _draw_action_greedy(self, state, policy_state):
        z = self._compute_phase(state, policy_state)

        mu = self._approximator(self._phi(z))

        next_policy_state = self.update_time(state, policy_state)

        return mu, next_policy_state

    def update_time(self, state, policy_state):
        """
        Method that updates the time counter. Can be overridden to introduce complex state-dependant behaviors.

        Args:
            state (np.ndarray): The current state of the system.

        """
        next_policy_state = policy_state + 1

        if not self._periodic:
            next_policy_state = np.minimum(next_policy_state, self._duration)

        return next_policy_state

    def _compute_phase(self, state, policy_state):
        """
        Method that updates the state variable. It can be overridden to implement state dependent phase.

        Args:
            state (np.ndarray): The current state of the system.

        Returns:
            The current value of the phase variable

        """
        return policy_state / self._duration

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def set_duration(self, duration):
        """
        Set the duration of the movement

        """
        assert duration >= 2
        self._duration = duration - 1

    def reset(self):
        self._policy_state = np.zeros(1)

        return self._policy_state

    def reset_vectorized(self, start_mask):
        if self._policy_state is None:
            self._policy_state = np.zeros((len(start_mask), 1))
        self._policy_state[start_mask] = 0.

        return self._policy_state
