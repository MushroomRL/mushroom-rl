import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter


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
            beta ([float, Parameter]): the temperature for the exponential reward
                transformation.

        """
        self._beta = to_parameter(beta)

        self._add_save_attr(_beta='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        Jep -= np.max(Jep)

        d = np.exp(self._beta() * Jep)

        self.distribution.mle(theta, d)
