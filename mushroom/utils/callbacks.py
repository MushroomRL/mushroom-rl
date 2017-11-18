from copy import deepcopy

import numpy as np

from mushroom.approximators import EnsembleTable


class CollectDataset:
    """
    This callback can be used to collect the samples during the learning of the
    agent.

    """
    def __init__(self):
        self._dataset = list()

    def __call__(self, **kwargs):
        self._dataset += kwargs['dataset']

    def get(self):
        return self._dataset


class CollectQ:
    """
    This callback can be used to collect the action values in a given state at
    each call.

    """
    def __init__(self, approximator):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use;

        """
        self._approximator = approximator

        self._qs = list()

    def __call__(self, **kwargs):
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._qs.append(deepcopy(np.mean(qs, 0)))
        else:
            self._qs.append(deepcopy(self._approximator.table))

    def get_values(self):
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
            approximator (object): the approximator to use;
            state (np.array): the state to consider.

        """
        self._approximator = approximator
        self._state = state

        self._max_qs = list()

    def __call__(self, **kwargs):
        q = self._approximator.predict(self._state)
        max_q = np.max(q)

        self._max_qs.append(max_q)

    def get_values(self):
        return self._max_qs


class CollectParameters:
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, parameter, *idx):
        self._parameter = parameter
        self._idx = idx
        self._p = list()

    def __call__(self, **kwargs):
        value = self._parameter.get_value(*self._idx)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):
        return self._p
