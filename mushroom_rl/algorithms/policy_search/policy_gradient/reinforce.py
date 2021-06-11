import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class REINFORCE(PolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning", Williams R. J.. 1992.

    """
    def __init__(self, mdp_info, policy, optimizer, features=None):
        super().__init__(mdp_info, policy, optimizer, features)
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.baseline_num = list()
        self.baseline_den = list()

        self._add_save_attr(
            sum_d_log_pi='numpy',
            list_sum_d_log_pi='pickle',
            baseline_num='pickle',
            baseline_den='pickle'
        )

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num, axis=0) / np.mean(self.baseline_den, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.
        grad_J_episode = list()
        for i, J_episode in enumerate(J):
            sum_d_log_pi = self.list_sum_d_log_pi[i]
            grad_J_episode.append(sum_d_log_pi * (J_episode - baseline))

        grad_J = np.mean(grad_J_episode, axis=0)
        self.list_sum_d_log_pi = list()
        self.baseline_den = list()
        self.baseline_num = list()

        return grad_J

    def _step_update(self, x, u, r):
        d_log_pi = self.policy.diff_log(x, u)
        self.sum_d_log_pi += d_log_pi

    def _episode_end_update(self):
        self.list_sum_d_log_pi.append(self.sum_d_log_pi)
        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)
        self.baseline_num.append(squared_sum_d_log_pi * self.J_episode)
        self.baseline_den.append(squared_sum_d_log_pi)

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)
