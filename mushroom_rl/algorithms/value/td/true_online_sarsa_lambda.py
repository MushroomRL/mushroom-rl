import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import get_action_features
from mushroom_rl.rl_utils.parameters import to_parameter


class TrueOnlineSARSALambda(TD):
    """
    True Online SARSA(lambda) with linear function approximation.
    "True Online TD(lambda)". Seijen H. V. et al.. 2014.

    """
    def __init__(self, mdp_info, policy, learning_rate, lambda_coeff, approximator_params=None):
        """
        Constructor.

        Args:
            lambda_coeff ([float, Parameter]): eligibility trace coefficient.

        """
        approximator_params = dict() if approximator_params is None else approximator_params

        Q = Regressor(LinearApproximator, **approximator_params)
        self.e = np.zeros(Q.weights_size)
        self._lambda = to_parameter(lambda_coeff)
        self._q_old = None

        self._add_save_attr(
            _q_old='numpy',
            _lambda='mushroom',
            e='numpy'
        )

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        phi_state = self.Q.model.phi(state)
        phi_state_action = get_action_features(phi_state, action, self.mdp_info.action_space.n)
        q_current = self.Q.predict(state, action)

        if self._q_old is None:
            self._q_old = q_current

        alpha = self._alpha(state, action)

        e_phi = self.e.dot(phi_state_action)
        self.e = (self.mdp_info.gamma * self._lambda() * self.e +
                  alpha * (1. - self.mdp_info.gamma * self._lambda.get_value() * e_phi) * phi_state_action)

        self.next_action, _ = self.draw_action(next_state)
        q_next = self.Q.predict(next_state, self.next_action) if not absorbing else 0.

        delta = reward + self.mdp_info.gamma * q_next - self._q_old

        theta = self.Q.get_weights()
        theta += delta * self.e + alpha * (self._q_old - q_current) * phi_state_action
        self.Q.set_weights(theta)

        self._q_old = q_next

    def episode_start(self, initial_state, episode_info):
        self._q_old = None
        self.e = np.zeros(self.Q.weights_size)

        return super().episode_start(initial_state, episode_info)
