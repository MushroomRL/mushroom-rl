import numpy as np
from copy import deepcopy

from mushroom_rl.utils.callbacks.callback import CallbackList
from mushroom_rl.approximators import Ensemble


class CollectQ(CallbackList):
    """
    This callback can be used to collect the action values in all states at the
    current time step.

    """
    def __init__(self, approximator):
        """
        Constructor.

        Args:
            approximator ([Table, Ensemble]): the approximator to use to
                predict the action values.

        """
        self._approximator = approximator

        super().__init__()

    def __call__(self, dataset):
        if isinstance(self._approximator, Ensemble):
            qs = list()
            for m in self._approximator:
                qs.append(m.table)
            self._data_list.append(deepcopy(np.mean(qs, 0)))
        else:
            self._data_list.append(deepcopy(self._approximator.table))
