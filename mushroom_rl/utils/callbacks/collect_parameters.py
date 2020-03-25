from mushroom_rl.utils.callbacks.callback import Callback
import numpy as np


class CollectParameters(Callback):
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, parameter, *idx):
        """
        Constructor.

        Args:
            parameter (Parameter): the parameter whose values have to be
                collected;
            *idx (list): index of the parameter when the ``parameter`` is
                tabular.

        """
        self._parameter = parameter
        self._idx = idx

        super().__init__()

    def __call__(self, dataset):
        value = self._parameter.get_value(*self._idx)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._data_list.append(value)
