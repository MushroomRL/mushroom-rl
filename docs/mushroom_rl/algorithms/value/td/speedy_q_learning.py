import numpy as np
from copy import deepcopy

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.table import Table


class SpeedyQLearning(TD):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.

    """
    def __init__(self, mdp_info, policy, learning_rate):
        Q = Table(mdp_info.size)
        self.old_q = deepcopy(Q)

        self._add_save_attr(old_q='mushroom')

        super().__init__(mdp_info, policy, Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        old_q = deepcopy(self.Q)

        max_q_cur = np.max(self.Q[next_state, :]) if not absorbing else 0.
        max_q_old = np.max(self.old_q[next_state, :]) if not absorbing else 0.

        target_cur = reward + self.mdp_info.gamma * max_q_cur
        target_old = reward + self.mdp_info.gamma * max_q_old

        alpha = self._alpha(state, action)
        q_cur = self.Q[state, action]
        self.Q[state, action] = q_cur + alpha * (target_old - q_cur) + (
            1. - alpha) * (target_cur - target_old)

        self.old_q = old_q
