from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor


class BatchTD(Agent):
    """
    Abstract class to implement a generic Batch TD algorithm.

    """
    def __init__(self, mdp_info, policy, approximator, approximator_params=None, fit_params=None):
        """
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.
            approximator_params (dict, None): parameters of the approximator to
                build;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;

        """
        approximator_params = dict() if approximator_params is None else\
            approximator_params
        self._fit_params = dict() if fit_params is None else fit_params

        self.approximator = Regressor(approximator, **approximator_params)
        policy.set_q(self.approximator)

        self._add_save_attr(
            approximator='mushroom',
            _fit_params='pickle'
        )

        super().__init__(mdp_info, policy)

    def _post_load(self):
        self.policy.set_q(self.approximator)
