from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor


class BatchTD(Agent):
    """
    Abstract class to implement a generic Batch TD algorithm.

    """
    def __init__(self, approximator, policy, mdp_info, fit_params=None,
                 approximator_params=None, features=None):
        """
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            approximator_params (dict, None): parameters of the approximator to
                build;

        """
        self._fit_params = dict() if fit_params is None else fit_params
        self._approximator_params = dict() if approximator_params is None else\
            approximator_params

        self.approximator = Regressor(approximator,
                                      **self._approximator_params)
        policy.set_q(self.approximator)

        super().__init__(policy, mdp_info, features)
