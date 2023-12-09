import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators import Regressor
from mushroom_rl.rl_utils.parameters import to_parameter


class SARSALambdaContinuous(TD):
    """
    Continuous version of SARSA(lambda) algorithm.

    """
    def __init__(self, mdp_info, policy, approximator, learning_rate, lambda_coeff, approximator_params=None):
        """
        Constructor.

        Args:
            lambda_coeff ([float, Parameter]): eligibility trace coefficient.

        """
        approximator_params = dict() if approximator_params is None else approximator_params

        Q = Regressor(approximator, **approximator_params)
        self.e = np.zeros(Q.weights_size)
        self._lambda = to_parameter(lambda_coeff)

        self._add_save_attr(
            _lambda='primitive',
            e='numpy'
        )

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q.predict(state, action)

        alpha = self._alpha(state, action)

        self.e = self.mdp_info.gamma * self._lambda() * self.e + self.Q.diff(state, action)

        self.next_action, _ = self.draw_action(next_state)
        q_next = self.Q.predict(next_state, self.next_action) if not absorbing else 0.

        delta = reward + self.mdp_info.gamma * q_next - q_current

        theta = self.Q.get_weights()
        theta += alpha * delta * self.e
        self.Q.set_weights(theta)

    def episode_start(self, initial_state, episode_info):
        self.e = np.zeros(self.Q.weights_size)

        return super().episode_start(initial_state, episode_info)
