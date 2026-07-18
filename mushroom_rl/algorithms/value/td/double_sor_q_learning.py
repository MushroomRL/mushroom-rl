import numpy as np
from copy import deepcopy

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.approximators.table import Table


class DoubleSORQLearning(TD):
    """
    Double Successive Over-Relaxation Q-Learning algorithm.
    "Double Successive Over-Relaxation Q-Learning with an Extension to Deep
    Reinforcement Learning".
    Shreyas S. R. 2025.

    """
    def __init__(self, mdp_info, policy, learning_rate, relaxation_factor):
        """
        Constructor.

        Args:
            relaxation_factor (float): the SOR relaxation factor. A value of
                one recovers Double Q-Learning.

        """
        if relaxation_factor <= 0.:
            raise ValueError('The relaxation factor must be positive.')

        self._relaxation_factor = relaxation_factor
        Q = Table(n_models=2, shape=mdp_info.size)

        super().__init__(mdp_info, policy, Q, learning_rate)

        self._alpha_double = [deepcopy(self._alpha), deepcopy(self._alpha)]

        self._add_save_attr(
            _alpha_double='primitive',
            _relaxation_factor='primitive'
        )

    def _update(self, state, action, reward, next_state, absorbing):
        approximator_idx = 0 if np.random.uniform() < .5 else 1
        other_idx = 1 - approximator_idx

        q_current = self.Q[approximator_idx][state, action]
        current_action = self._greedy_action(
            self.Q[approximator_idx][state, :])
        q_relaxation = self.Q[other_idx][state, current_action]

        if not absorbing:
            next_action = self._greedy_action(
                self.Q[approximator_idx][next_state, :])
            q_next = self.Q[other_idx][next_state, next_action]
        else:
            q_next = 0.

        w = self._relaxation_factor
        q_target = w * (reward + self.mdp_info.gamma * q_next) + \
            (1. - w) * q_relaxation
        q = q_current + self._alpha_double[approximator_idx](state, action) * \
            (q_target - q_current)

        self.Q[approximator_idx][state, action] = q

    @staticmethod
    def _greedy_action(q):
        max_q = np.max(q)
        return np.array([np.random.choice(np.argwhere(q == max_q).ravel())])
