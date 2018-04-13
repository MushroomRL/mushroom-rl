from copy import deepcopy

import numpy as np

from mushroom.utils.table import EnsembleTable


class CollectDataset:
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """
    def __init__(self):
        """
        Constructor.

        """
        self._dataset = list()

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        self._dataset += dataset

    def get(self):
        """
        Returns:
             The current samples list.

        """
        return self._dataset

    def clean(self):
        """
        Deletes the current dataset

        """
        self._dataset = list()


class CollectQ:
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

        self._qs = list()

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
            self._qs.append(deepcopy(np.mean(qs, 0)))
        else:
            self._qs.append(deepcopy(self._approximator.table))

    def get_values(self):
        """
        Returns:
             The current action-values list.

        """
        return self._qs


class CollectMaxQ:
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

        self._max_qs = list()

    def __call__(self, **kwargs):
        """
        Add maximum action values to the maximum action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        q = self._approximator.predict(self._state)
        max_q = np.max(q)

        self._max_qs.append(max_q)

    def get_values(self):
        """
        Returns:
             The current maximum action-values list.

        """
        return self._max_qs


class CollectParameters:
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
        self._p = list()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        value = self._parameter.get_value(*self._idx)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):
        """
        Returns:
             The current parameter values list.

        """
        return self._p
