import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class eNAC(PolicyGradient):
    """
    Episodic Natural Actor Critic algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J. 2013.

    """
    def __init__(self, mdp_info, policy, optimizer, features=None,
                 critic_features=None):
        """
        Constructor.

        Args:
            critic_features (Features, None): features used by the critic.

        """
        super().__init__(mdp_info, policy, optimizer, features)
        self.phi_c = critic_features

        self.sum_grad_log = None
        self.psi_ext = None
        self.sum_grad_log_list = list()

        self._add_save_attr(
            phi_c='pickle', 
            sum_grad_log='numpy', 
            psi_ext='pickle', 
            sum_grad_log_list='pickle'
        )

    def _compute_gradient(self, J):
        R = np.array(J)
        PSI = np.array(self.sum_grad_log_list)

        w_and_v = np.linalg.pinv(PSI).dot(R)
        nat_grad = w_and_v[:self.policy.weights_size]

        self.sum_grad_log_list = list()

        return nat_grad

    def _step_update(self, x, u, r):
        self.sum_grad_log += self.df*self.policy.diff_log(x, u)

        if self.psi_ext is None:
            if self.phi_c is None:
                self.psi_ext = np.ones(1)
            else:
                self.psi_ext = self.phi_c(x)

    def _episode_end_update(self):
        psi = np.concatenate((self.sum_grad_log, self.psi_ext))
        self.sum_grad_log_list.append(psi)

    def _init_update(self):
        self.psi_ext = None
        self.sum_grad_log = np.zeros(self.policy.weights_size)
