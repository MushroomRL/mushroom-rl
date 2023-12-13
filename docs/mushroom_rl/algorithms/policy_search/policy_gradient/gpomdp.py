import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class GPOMDP(PolicyGradient):
    """
    GPOMDP algorithm.
    "Infinite-Horizon Policy-Gradient Estimation". Baxter J. and Bartlett P. L..
    2001.

    """
    def __init__(self, mdp_info, policy, optimizer):
        super().__init__(mdp_info, policy, optimizer)

        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.list_sum_d_log_pi_ep = list()

        self.list_reward = list()
        self.list_reward_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        self.step_count = 0

        self._add_save_attr(
            sum_d_log_pi='numpy',
            list_sum_d_log_pi='pickle',
            list_sum_d_log_pi_ep='pickle',
            list_reward='pickle',
            list_reward_ep='pickle',
            baseline_num='pickle',
            baseline_den='pickle',
            step_count='numpy'
        )

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def _compute_gradient(self, J):
        n_episodes = len(self.list_sum_d_log_pi_ep)
        grad_J_episode = list()
        for i in range(n_episodes):
            list_sum_d_log_pi = self.list_sum_d_log_pi_ep[i]
            list_reward = self.list_reward_ep[i]

            n_steps = len(list_sum_d_log_pi)

            gradient = np.zeros(self.policy.weights_size)

            for t in range(n_steps):
                step_grad = list_sum_d_log_pi[t]
                step_reward = list_reward[t]
                baseline = np.mean(self.baseline_num[t], axis=0) / np.mean(self.baseline_den[t], axis=0)
                baseline[np.logical_not(np.isfinite(baseline))] = 0.
                gradient += step_grad * (step_reward - baseline)

            grad_J_episode.append(gradient)

        gradJ = np.mean(grad_J_episode, axis=0)

        self.list_reward_ep = list()
        self.list_sum_d_log_pi_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        return gradJ

    def _step_update(self, x, u, r):
        discounted_reward = self.df * r
        self.list_reward.append(discounted_reward)

        d_log_pi = self.policy.diff_log(x, u)
        self.sum_d_log_pi += d_log_pi

        self.list_sum_d_log_pi.append(self.sum_d_log_pi.copy())

        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)

        if self.step_count >= len(self.baseline_num):
            self.baseline_num.append(list())
            self.baseline_den.append(list())

        self.baseline_num[self.step_count].append(discounted_reward * squared_sum_d_log_pi)
        self.baseline_den[self.step_count].append(squared_sum_d_log_pi)

        self.step_count += 1

    def _episode_end_update(self):
        self.list_reward_ep.append(self.list_reward)
        self.list_reward = list()

        self.list_sum_d_log_pi_ep.append(self.list_sum_d_log_pi)
        self.list_sum_d_log_pi = list()

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)
        self.list_sum_d_log_pi = list()
        self.step_count = 0
