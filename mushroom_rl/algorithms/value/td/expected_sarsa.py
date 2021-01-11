from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table


class ExpectedSARSA(TD):
    """
    Expected SARSA algorithm.
    "A theoretical and empirical analysis of Expected Sarsa". Seijen H. V. et
    al.. 2009.

    """
    def __init__(self, mdp_info, policy, learning_rate):
        Q = Table(mdp_info.size)

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        if not absorbing:
            q_next = self.Q[next_state, :].dot(self.policy(next_state))
        else:
            q_next = 0.

        self.Q[state, action] = q_current + self._alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)
