import numpy as np
from scipy.stats import multivariate_normal

from .policy import ParametricPolicy


class ProMP(ParametricPolicy):
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
        self._sigma = sigma
        self._periodic = periodic

        self._add_save_attr(
            _approximator='mushroom',
            _phi='mushroom',
            _duration='primitive',
            _sigma='numpy',
            _periodic='primitive'
        )

    def __call__(self, state, action):
        z = self._compute_phase(state)
        mu = self._approximator(self._phi(z))

        if self._sigma is None:
            return 1.0 if mu == action else 0.0
        else:
            return multivariate_normal.pdf(action, mu, self._sigma)

    def draw_action(self, state, policy_state):
        z = self._compute_phase(state)

        mu = self._approximator(self._phi(z))

        next_policy_state = self.update_time(state, policy_state)

        if self._sigma is None:
            return mu, next_policy_state
        else:
            return np.random.multivariate_normal(mu, self._sigma), next_policy_state

    def update_time(self, state, policy_state):
        """
        Method that updates the time counter. Can be overridden to introduce complex state-dependant behaviors.

        Args:
            state (np.ndarray): The current state of the system.

        """
        policy_state += 1

        if not self._periodic and policy_state >= self._duration:
            policy_state = self._duration

        return policy_state

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
        return 0
