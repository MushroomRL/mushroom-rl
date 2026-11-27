import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class REINFORCE(PolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
    Williams R. J. 1992.

    """
    def _compute_gradient(self, dataset):
        J = dataset.compute_J(self.mdp_info.gamma).reshape(-1, 1)
        sum_d_log_pi = np.sum(self._d_log_pi(dataset), axis=1)

        divisor = np.sum(sum_d_log_pi ** 2, axis=0)
        zero_divisor = divisor == 0.
        divisor[zero_divisor] = 1.
        baseline = np.sum(J * sum_d_log_pi ** 2, axis=0) / divisor
        baseline[zero_divisor] = 0.

        return np.mean(sum_d_log_pi * (J - baseline), axis=0)
