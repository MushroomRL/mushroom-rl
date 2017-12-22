import numpy as np
from mushroom.algorithms.policy_search import PolicyGradient


class eNAC(PolicyGradient):
    """
    ENAC algorithm.
    "Policy Gradient Methods for Robotics", Peters J., Schaal S.  2006.

    """
    def __init__(self, policy, mdp_info, params, features=None):
        self.__name__ = 'eNAC'

        super(eNAC, self).__init__(policy, mdp_info, params, features)

        self.psi = None
        self.fisher_list = list()
        self.grad_list = list()
        self.eligibility_list = list()

    def _compute_gradient(self, J):
        n_ep = len(self.fisher_list)

        fisher = np.mean(self.fisher_list, axis=0)
        g = np.mean(self.grad_list, axis=0)
        eligibility = np.mean(self.eligibility_list, axis=0)
        J_pol = np.mean(J)

        if fisher.shape[0] == np.linalg.matrix_rank(fisher):
            tmp = np.linalg.solve(
                n_ep * fisher - np.outer(eligibility, eligibility), eligibility)

            print eligibility
            print tmp
            Q = (1 + eligibility.dot(tmp)) / n_ep
            if type(Q) is not np.ndarray:
                Q = np.array([Q])
            b = Q.dot(J_pol - eligibility.dot(np.linalg.solve(fisher, g)))
            gradient = g - eligibility.dot(b)
            nat_grad = np.linalg.solve(fisher, gradient)
        else:
            H = np.linalg.pinv(fisher)
            b = (1 + eligibility.dot(np.linalg.pinv(n_ep * fisher - np.outer(
                eligibility, eligibility)).dot(eligibility))) * (
                    J_pol - eligibility.dot(H).dot(g)) / n_ep
            gradient = g - eligibility.dot(b)
            nat_grad = H.dot(gradient)

        self.fisher_list = list()
        self.grad_list = list()
        self.eligibility_list = list()

        return gradient, nat_grad

    def _step_update(self, x, u, r):
        self.psi += self.policy.diff_log(x, u)

    def _episode_end_update(self):
        f_m = np.outer(self.psi, self.psi)
        self.fisher_list.append(f_m)

        gradient = self.J_episode * self.psi
        self.grad_list.append(gradient)

        self.eligibility_list.append(self.psi)

    def _init_update(self):
        self.psi = np.zeros(self.policy.weights_size)
