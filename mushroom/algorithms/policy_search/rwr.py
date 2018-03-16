from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import compute_J
import numpy as np


class RWR(Agent):
    def __init__(self, distribution, policy, mdp_info, beta, features=None):
        """
        Constructor.

        Args:
            distribution: the distribution of policy parameters
            policy: the policy to use
            beta (float): the temperature for the exponential
             reward transformation.

        """
        self.distribution = distribution
        self.beta = beta
        self._theta_list = list()

        super().__init__(policy, mdp_info, features)


    def episode_start(self):
        theta = self.distribution.sample()
        self._theta_list.append(theta)
        self.policy.set_weights(theta)


    def fit(self, dataset):
        Jep = compute_J(dataset, self.mdp_info.gamma)

        Jep = np.array(Jep)

        # for numerical stability
        max_J = np.max(Jep)
        Jep -= max_J

        d = np.exp(self.beta*Jep)
        theta = np.array(self._theta_list)

        self.distribution.mle(theta, d)

        self._theta_list = list()

    def stop(self):
        self._theta_list = list()