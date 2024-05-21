import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.table import Table


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.

    """
    def __init__(self, mdp_info, policy, learning_rate):
        Q = Table(mdp_info.size)

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.

        self.Q[state, action] = q_current + self._alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)
