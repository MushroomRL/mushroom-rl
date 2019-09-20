from mushroom.algorithms.value.td import TD
from mushroom.utils.eligibility_trace import EligibilityTrace
from mushroom.utils.table import Table


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, policy, mdp_info, learning_rate):
        self.Q = Table(mdp_info.size)
        super().__init__(self.Q, policy, mdp_info, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        self.next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

        self.Q[state, action] = q_current + self.alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)


class SARSALambda(TD):
    """
    The SARSA(lambda) algorithm for finite MDPs.

    """
    def __init__(self, policy, mdp_info, learning_rate, lambda_coeff,
                 trace='replacing'):
        """
        Constructor.

        Args:
            lambda_coeff (float): eligibility trace coefficient;
            trace (str, 'replacing'): type of eligibility trace to use.

        """
        self.Q = Table(mdp_info.size)
        self._lambda = lambda_coeff

        self.e = EligibilityTrace(self.Q.shape, trace)
        super().__init__(self.Q, policy, mdp_info, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        self.next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

        delta = reward + self.mdp_info.gamma * q_next - q_current
        self.e.update(state, action)

        self.Q.table += self.alpha(state, action) * delta * self.e.table
        self.e.table *= self.mdp_info.gamma * self._lambda

    def episode_start(self):
        self.e.reset()

        super().episode_start()


class ExpectedSARSA(TD):
    """
    Expected SARSA algorithm.
    "A theoretical and empirical analysis of Expected Sarsa". Seijen H. V. et
    al.. 2009.

    """
    def __init__(self, policy, mdp_info, learning_rate):
        self.Q = Table(mdp_info.size)
        super().__init__(self.Q, policy, mdp_info, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        if not absorbing:
            q_next = self.Q[next_state, :].dot(self.policy(next_state))
        else:
            q_next = 0.

        self.Q[state, action] = q_current + self.alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)