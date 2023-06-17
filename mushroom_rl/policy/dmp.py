import numpy as np

from .policy import ParametricPolicy


class DMP(ParametricPolicy):
    """
    Class representing a Dynamic Movement Primitive (DMP).

    Differently from the original implementation of DMP, an arbitrary regressor can be used to compute the mean from
    phase variable.

    """
    def __init__(self, mu, phi, goal, dt, tau, alpha_v, beta_v, alpha_z, beta_z):
        self._approximator = mu
        self._phi = phi

        self._g = goal

        self._dt = dt

        self._tau = tau
        self._alpha_v = alpha_v
        self._beta_v = beta_v
        self._alpha_z = alpha_z
        self._beta_z = beta_z

        action_size = self._approximator.output_shape

        self._v = np.zeros(action_size)
        self._x = np.zeros(action_size)
        self._z = np.zeros(action_size)
        self._y = np.zeros(action_size)

        self._add_save_attr(
            _approximator='mushroom',
            _phi='mushroom',
            _dt='primitive',
            _tau='primitive',
            _alpha_v='numpy',
            _beta_v='numpy',
            _alpha_z='numpy',
            _beta_z='numpy',
            _g='numpy',
            _v='numpy',
            _x='numpy',
            _z='numpy',
            _y='numpy'
        )

    def __call__(self, state, action):
        return 1.0 if np.allclose(self._y == action) else 0.0

    def draw_action(self, state):
        self.update_system(state)

        return self._y

    def set_goal(self, goal):
        self._g = goal

    def get_goal(self, state):
        return self._g

    def update_system(self, state):
        """
        Method that updates the dynamical system. Can be overridden to introduce complex state-dependant behaviors.

        Args:
            state (np.ndarray): The current state of the system.

        """
        g = self.get_goal(state)

        f = self._approximator(self._phi(self._x/g)) * self._v

        v_dot, x_dot = self._canonical_system(g)
        y_dot, z_dot = self._transformation_system(f, g)

        self._v += v_dot * self._dt
        self._x += x_dot * self._dt
        self._z += z_dot * self._dt
        self._y += y_dot * self._dt

    def _transformation_system(self, f, g):
        z_dot = self._alpha_z * (self._beta_z * (g - self._y) - self._z) / self._tau
        y_dot = (self._z + f) / self._tau

        return y_dot, z_dot

    def _canonical_system(self, g):
        v_dot = self._alpha_v * (self._beta_v * (g - self._y) - self._v) / self._tau
        x_dot = self._v / self._tau

        return v_dot, x_dot

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        action_size = self._approximator.output_shape
        self._v = np.zeros(action_size)
        self._x = np.zeros(action_size)
        self._z = np.zeros(action_size)
        self._y = np.zeros(action_size)

