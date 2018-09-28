import numpy as np

from .policy import ParametricPolicy


class OrnsteinUhlenbeckPolicy(ParametricPolicy):
    """
    Ornstein-Uhlenbeck process as implemented in:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py.

    This policy is commonly used in the Deep Deterministic Policy Gradient
    algorithm.

    """
    def __init__(self, mu, sigma, theta, dt, x0=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            sigma (np.ndarray): average magnitude of the random flactations per
                square-root time;
            theta (float): rate of mean reversion;
            dt (float): time interval;
            x0 (np.ndarray, None): initial values of noise.

        """
        self._approximator = mu
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._x0 = x0

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        mu = self._approximator.predict(state)

        x = self.x_prev - self._theta * self.x_prev * self._dt + self._sigma *\
            np.sqrt(self._dt) * np.random.normal(size=self._approximator.output_shape)
        self.x_prev = x

        return mu + x

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        self.x_prev = self._x0 if self._x0 is not None else np.zeros(self._approximator.output_shape)
