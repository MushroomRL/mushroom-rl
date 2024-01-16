import numpy as np
from copy import deepcopy

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.ensemble_table import EnsembleTable


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". Hasselt H. V.. 2010.

    """
    def __init__(self, mdp_info, policy, learning_rate):
        Q = EnsembleTable(2, mdp_info.size)

        super().__init__(mdp_info, policy, Q, learning_rate)

        self._alpha_double = [deepcopy(self._alpha), deepcopy(self._alpha)]

        self._add_save_attr(
            _alpha_double='primitive'
        )

        assert len(self.Q) == 2, 'The regressor ensemble must' \
                                 ' have exactly 2 models.'

    def _update(self, state, action, reward, next_state, absorbing):
        approximator_idx = 0 if np.random.uniform() < .5 else 1

        q_current = self.Q[approximator_idx][state, action]

        if not absorbing:
            q_ss = self.Q[approximator_idx][next_state, :]
            max_q = np.max(q_ss)
            a_n = np.array(
                [np.random.choice(np.argwhere(q_ss == max_q).ravel())])
            q_next = self.Q[1 - approximator_idx][next_state, a_n]
        else:
            q_next = 0.

        q = q_current + self._alpha_double[approximator_idx](state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)

        self.Q[approximator_idx][state, action] = q
