import numpy as np

from mushroom.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class RWR(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, distribution, policy, mdp_info, beta, features=None):
        """
        Constructor.

        Args:
            beta (float): the temperature for the exponential reward
                transformation.

        """
        self.beta = beta

        super().__init__(distribution, policy, mdp_info, features)

    def _update(self, Jep, theta):
        Jep -= np.max(Jep)

        d = np.exp(self.beta * Jep)

        self.distribution.mle(theta, d)