import numpy as np
from copy import deepcopy

from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.utils.table import EnsembleTable


class CollectQ(Callback):
    """
    This callback can be used to collect the action values in all states at the
    current time step.

    """
    def __init__(self, approximator):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use to
                predict the action values.

        """
        self._approximator = approximator

        super().__init__()

    def __call__(self, dataset):
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._data_list.append(deepcopy(np.mean(qs, 0)))
        else:
            self._data_list.append(deepcopy(self._approximator.table))
