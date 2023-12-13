from mushroom_rl.utils.callbacks.callback import CallbackList
import numpy as np


class CollectMaxQ(CallbackList):
    """
    This callback can be used to collect the maximum action value in a given
    state at each call.

    """
    def __init__(self, approximator, state):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use;
            state (np.ndarray): the state to consider.

        """
        self._approximator = approximator
        self._state = state

        super().__init__()

    def __call__(self, dataset):
        q = self._approximator.predict(self._state)
        max_q = np.max(q)

        self._data_list.append(max_q)
