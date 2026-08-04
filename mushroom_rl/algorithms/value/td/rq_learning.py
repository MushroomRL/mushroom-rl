import numpy as np

from mushroom_rl.core import HasNextAction
from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.table import Table
from mushroom_rl.rl_utils.parameters import Parameter


class RQLearning(TD):
    """
    RQ-Learning algorithm, off-policy version.
    "Exploiting Structure and Uncertainty of Bellman Updates in Markov Decision Processes".
    Tateo D. et al. 2017.

    """
    def __init__(self, mdp_info, policy, learning_rate, beta=None, delta=None):
        """
        Constructor.

        Args:
            beta ([float, Parameter], None): beta coefficient;
            delta ([float, Parameter], None): delta coefficient.

        """
        if delta is not None and beta is None:
            self.delta = Parameter.make(delta)
            self.beta = None
        elif delta is None and beta is not None:
            self.delta = None
            self.beta = Parameter.make(beta)
        else:
            raise ValueError('delta or beta parameters needed.')

        Q = Table(mdp_info.size)
        self.Q_tilde = Table(mdp_info.size)
        self.R_tilde = Table(mdp_info.size)

        self._add_save_attr(
            delta='mushroom',
            beta='mushroom',
            Q_tilde='mushroom',
            R_tilde='mushroom'
        )

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        alpha = self._alpha(state, action, target=reward)
        self.R_tilde[state, action] += alpha * (reward - self.R_tilde[state, action])

        if not absorbing:
            q_next = self._next_q(next_state)

            if self.delta is not None:
                beta = alpha * self.delta(state, action, target=q_next, factor=alpha)
            else:
                beta = self.beta(state, action, target=q_next)

            self.Q_tilde[state, action] += beta * (q_next - self.Q_tilde[state, action])

        self.Q[state, action] = self.R_tilde[state, action] + self.mdp_info.gamma * self.Q_tilde[state, action]

    def _next_q(self, next_state):
        """
        Args:
            next_state (np.ndarray): the state where next action has to be
                evaluated.

        Returns:
            The weighted estimator value in 'next_state'.

        """
        return np.max(self.Q[next_state, :])


class RQLearningOnPolicy(HasNextAction, RQLearning):
    """
    RQ-Learning algorithm, on-policy version. The next-state value is estimated on the action drawn by the policy
    (cached and executed at the next step) instead of the greedy one.

    """
    def _next_q(self, next_state):
        self._next_action = self.draw_action(next_state)

        return self.Q[next_state, self._next_action]
