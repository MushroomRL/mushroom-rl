import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class RWR(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, beta, features=None):
        """
        Constructor.

        Args:
            beta (float): the temperature for the exponential reward
                transformation.

        """
        self.beta = beta

        self._add_save_attr(beta='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        Jep -= np.max(Jep)

        d = np.exp(self.beta * Jep)

        self.distribution.mle(theta, d)
