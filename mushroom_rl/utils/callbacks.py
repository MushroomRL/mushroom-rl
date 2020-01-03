from copy import deepcopy

import numpy as np

from mushroom_rl.utils.table import EnsembleTable


class Callback(object):
    """
    Interface for all basic callbacks. Implements a list in which it is possible to store data and
    methods to query and clean the content stored by the callback.


    """
    def __init__(self):
        self._data_list = list()

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        raise NotImplementedError

    def get(self):
        """
        Returns:
             The current collected data as a list.

        """
        return self._data_list

    def clean(self):
        """
        Deletes the current stored data list

        """
        self._data_list = list()


class CollectDataset(Callback):
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """
    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        self._data_list += dataset


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

    def __call__(self, **kwargs):
        """
        Add action values to the action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._data_list.append(deepcopy(np.mean(qs, 0)))
        else:
            self._data_list.append(deepcopy(self._approximator.table))


class CollectMaxQ(Callback):
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

    def __call__(self, **kwargs):
        """
        Add maximum action values to the maximum action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        q = self._approximator.predict(self._state)
        max_q = np.max(q)

        self._data_list.append(max_q)


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

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        value = self._parameter.get_value(*self._idx)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._data_list.append(value)
