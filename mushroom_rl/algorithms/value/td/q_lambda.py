import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.eligibility_trace import EligibilityTrace
from mushroom_rl.utils.table import Table
from mushroom_rl.utils.parameters import to_parameter


class QLambda(TD):
    """
    Q(Lambda) algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.

    """
    def __init__(self, mdp_info, policy, learning_rate, lambda_coeff,
                 trace='replacing'):
        """
        Constructor.

        Args:
            lambda_coeff ([float, Parameter]): eligibility trace coefficient;
            trace (str, 'replacing'): type of eligibility trace to use.

        """
        Q = Table(mdp_info.size)
        self._lambda = to_parameter(lambda_coeff)

        self.e = EligibilityTrace(Q.shape, trace)
        self._add_save_attr(
            _lambda='mushroom',
            e='mushroom'
        )

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.

        delta = reward + self.mdp_info.gamma*q_next - q_current
        self.e.update(state, action)

        self.Q.table += self._alpha(state, action) * delta * self.e.table
        self.e.table *= self.mdp_info.gamma * self._lambda()

    def episode_start(self):
        self.e.reset()

        super().episode_start()
