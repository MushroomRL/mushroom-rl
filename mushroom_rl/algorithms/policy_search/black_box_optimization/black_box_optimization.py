import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.utils.dataset import compute_J


class BlackBoxOptimization(Agent):
    """
    Base class for black box optimization algorithms.
    These algorithms work on a distribution of policy parameters and often they
    do not rely on stochastic and differentiable policies.

    """
    def __init__(self, mdp_info, distribution, policy, features=None):
        """
        Constructor.

        Args:
            distribution (Distribution): the distribution of policy parameters;
            policy (ParametricPolicy): the policy to use.

        """
        self.distribution = distribution
        self._theta_list = list()

        self._add_save_attr(distribution='mushroom', _theta_list='pickle')

        super().__init__(mdp_info, policy, features)

    def episode_start(self):
        theta = self.distribution.sample()
        self._theta_list.append(theta)
        self.policy.set_weights(theta)

        super().episode_start()

    def fit(self, dataset, **info):
        Jep = compute_J(dataset, self.mdp_info.gamma)

        Jep = np.array(Jep)
        theta = np.array(self._theta_list)

        self._update(Jep, theta)

        self._theta_list = list()

    def stop(self):
        self._theta_list = list()

    def _update(self, Jep, theta):
        """
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the
        proper update.

        Args:
            Jep (np.ndarray): a vector containing the J of the considered
                trajectories;
            theta (np.ndarray): a matrix of policy parameters of the considered
                trajectories.

        """
        raise NotImplementedError('BlackBoxOptimization is an abstract class')
