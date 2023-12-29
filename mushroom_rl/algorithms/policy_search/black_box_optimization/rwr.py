import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.rl_utils.parameters import to_parameter


class RWR(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, beta):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the temperature for the exponential reward
                transformation.

        """
        assert not distribution.is_contextual

        self._beta = to_parameter(beta)

        super().__init__(mdp_info, distribution, policy)

        self._add_save_attr(_beta='mushroom')

    def _update(self, Jep, theta, context):
        Jep -= np.max(Jep)

        d = np.exp(self._beta() * Jep)

        self.distribution.mle(theta, d)
