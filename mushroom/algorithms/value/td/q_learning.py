import numpy as np
from copy import deepcopy

from mushroom.algorithms.value.td import TD
from mushroom.utils.table import EnsembleTable, Table


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.

    """
    def __init__(self, policy, mdp_info, learning_rate):
        self.Q = Table(mdp_info.size)

        super().__init__(self.Q, policy, mdp_info, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.

        self.Q[state, action] = q_current + self.alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". Hasselt H. V.. 2010.

    """
    def __init__(self, policy, mdp_info, learning_rate):
        self.Q = EnsembleTable(2, mdp_info.size)

        super().__init__(self.Q, policy, mdp_info, learning_rate)

        self.alpha = [deepcopy(self.alpha), deepcopy(self.alpha)]

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

        q = q_current + self.alpha[approximator_idx](state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)

        self.Q[approximator_idx][state, action] = q


class SpeedyQLearning(TD):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.

    """
    def __init__(self, policy, mdp_info, learning_rate):
        self.Q = Table(mdp_info.size)
        self.old_q = deepcopy(self.Q)

        super().__init__(self.Q, policy, mdp_info, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        old_q = deepcopy(self.Q)

        max_q_cur = np.max(self.Q[next_state, :]) if not absorbing else 0.
        max_q_old = np.max(self.old_q[next_state, :]) if not absorbing else 0.

        target_cur = reward + self.mdp_info.gamma * max_q_cur
        target_old = reward + self.mdp_info.gamma * max_q_old

        alpha = self.alpha(state, action)
        q_cur = self.Q[state, action]
        self.Q[state, action] = q_cur + alpha * (target_old - q_cur) + (
            1. - alpha) * (target_cur - target_old)

        self.old_q = old_q