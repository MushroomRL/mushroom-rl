import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.rl_utils.parameters import Parameter


class RWR(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics",
    Deisenroth M. P. et al. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, beta):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the inverse of the temperature of the exponential reward
                transformation. The higher it is, the more the update concentrates on the best episodes.

        """
        assert not distribution.is_contextual

        self._beta = Parameter.make(beta)

        super().__init__(mdp_info, distribution, policy)

        self._add_save_attr(_beta='mushroom')

    def _update(self, Jep, theta, context):
        Jep -= np.max(Jep)

        d = np.exp(self._beta() * Jep)

        self.distribution.mle(theta, d)
