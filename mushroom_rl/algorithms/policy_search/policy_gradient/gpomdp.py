import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient
from mushroom_rl.utils.episodes import split_episodes


class GPOMDP(PolicyGradient):
    """
    GPOMDP algorithm.
    "Infinite-Horizon Policy-Gradient Estimation".
    Baxter J. and Bartlett P. L. 2001.

    """
    def _compute_gradient(self, dataset):
        d_log_pi = self._d_log_pi(dataset)
        step = np.arange(d_log_pi.shape[1])
        episode_step_mask = (step < dataset.episodes_length[:, None])[:, :, None]

        reward = split_episodes(dataset.last, dataset.reward)[1]
        reward = (self.mdp_info.gamma ** step * reward)[:, :, None]

        sum_d_log_pi = np.cumsum(d_log_pi, axis=1) * episode_step_mask

        divisor = np.sum(sum_d_log_pi ** 2, axis=0)
        zero_divisor = divisor == 0.
        divisor[zero_divisor] = 1.
        baseline = np.sum(reward * sum_d_log_pi ** 2, axis=0) / divisor
        baseline[zero_divisor] = 0.

        return np.mean(np.sum(sum_d_log_pi * (reward - baseline), axis=1), axis=0)
