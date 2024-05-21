import numpy as np
from copy import deepcopy

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.ensemble_table import EnsembleTable


class MaxminQLearning(TD):
    """
    Maxmin Q-Learning algorithm without replay memory.
    "Maxmin Q-learning: Controlling the Estimation Bias of Q-learning".
    Lan Q. et al. 2019.

    """
    def __init__(self, mdp_info, policy, learning_rate, n_tables):
        """
        Constructor.

        Args:
            n_tables (int): number of tables in the ensemble.

        """
        self._n_tables = n_tables
        Q = EnsembleTable(n_tables, mdp_info.size, prediction='min')

        super().__init__(mdp_info, policy, Q, learning_rate)

        self._alpha_mm = [deepcopy(self._alpha) for _ in range(n_tables)]

        self._add_save_attr(_n_tables='primitive', _alpha_mm='primitive')

    def _update(self, state, action, reward, next_state, absorbing):
        approximator_idx = np.random.choice(self._n_tables)

        q_current = self.Q[approximator_idx][state, action]

        if not absorbing:
            q_ss = self.Q.predict(next_state)
            q_next = np.max(q_ss)
        else:
            q_next = 0.

        q = q_current + self._alpha_mm[approximator_idx](state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)

        self.Q[approximator_idx][state, action] = q
