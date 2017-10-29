import numpy as np
from mushroom.algorithms.policy_search import PolicyGradient


class REINFORCE(PolicyGradient):
    def __init__(self, policy, gamma, params, features):
        self.__name__ = 'REINFORCE'
        self.sumdlogpi = None
        self.list_sumdlogpi = []
        self.baseline_num = []
        self.baseline_den = []

        super(REINFORCE, self).__init__(policy, gamma, params, features)

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num)/np.mean(self.baseline_den)
        grad_Jep = []
        for i,Jep in enumerate(J):
            sumdlogpi = self.list_sumdlogpi[i]
            grad_Jep.append(sumdlogpi*(Jep-baseline))

        grad_J = np.mean(grad_Jep, axis=0)
        self.baseline_den = []
        self.baseline_num = []
        self.list_sumdlogpi = []
        return grad_J

    def _step_update(self, x, u):
        dlogpi = self.policy.diff_log(x, u)
        self.sumdlogpi += dlogpi

    def _episode_end_update(self, Jep):
        self.list_sumdlogpi.append(self.sumdlogpi)
        squared_sumdlogpi = np.square(self.sumdlogpi)
        self.baseline_num.append(squared_sumdlogpi*Jep)
        self.baseline_den.append(squared_sumdlogpi)

    def _init_update(self):
        self.sumdlogpi = np.zeros(self.policy.weights_shape)
