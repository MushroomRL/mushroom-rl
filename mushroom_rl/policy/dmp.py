import numpy as np

from mushroom_rl.policy.policy import ParametricPolicy


class DMP(ParametricPolicy):
    """
    Class representing a Dynamic Movement Primitive (DMP).

    Differently from the original implementation of DMP, an arbitrary regressor can be used to compute the mean from
    phase variable.

    The internal state of the dynamical system, i.e. the canonical velocity ``v``, the phase ``x``, and the
    transformation system variables ``z`` and ``y``, is stored in the policy state as a single array stacking the
    four variables, with shape ``(4,) + action_shape``.

    """
    def __init__(self, mu, phi, goal, dt, tau, alpha_v, beta_v, alpha_z, beta_z):
        self._action_shape = mu.output_shape

        super().__init__(policy_state_shape=(4,) + tuple(self._action_shape))

        self._approximator = mu
        self._phi = phi

        self._g = goal

        self._dt = dt

        self._tau = tau
        self._alpha_v = alpha_v
        self._beta_v = beta_v
        self._alpha_z = alpha_z
        self._beta_z = beta_z

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
            _action_shape='primitive'
        )

    def __call__(self, state, action, policy_state=None):
        next_policy_state = self.update_system(state, policy_state)
        y = next_policy_state[3]

        return 1.0 if np.allclose(y, action) else 0.0

    def draw_action(self, state, policy_state):
        next_policy_state = self.update_system(state, policy_state)
        y = next_policy_state[3]

        return y, next_policy_state

    def set_goal(self, goal):
        self._g = goal

    def get_goal(self, state):
        return self._g

    def update_system(self, state, policy_state):
        """
        Method that updates the dynamical system. Can be overridden to introduce complex state-dependant behaviors.

        Args:
            state (np.ndarray): the current state of the environment;
            policy_state (np.ndarray): the internal state of the DMP, stacking the ``[v, x, z, y]`` variables.

        Returns:
            The updated internal state of the DMP.

        """
        next_policy_state = np.array(policy_state)
        v, x, z, y = next_policy_state

        g = self.get_goal(state)

        f = self._approximator(self._phi(x / g)) * v

        v_dot, x_dot = self._canonical_system(g, v, y)
        y_dot, z_dot = self._transformation_system(f, g, y, z)

        v += v_dot * self._dt
        x += x_dot * self._dt
        z += z_dot * self._dt
        y += y_dot * self._dt

        return next_policy_state

    def _transformation_system(self, f, g, y, z):
        z_dot = self._alpha_z * (self._beta_z * (g - y) - z) / self._tau
        y_dot = (z + f) / self._tau

        return y_dot, z_dot

    def _canonical_system(self, g, v, y):
        v_dot = self._alpha_v * (self._beta_v * (g - y) - v) / self._tau
        x_dot = v / self._tau

        return v_dot, x_dot

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        return np.zeros((4,) + tuple(self._action_shape))