from copy import deepcopy

import numpy as np

from PyPi.approximators.ensemble import Ensemble
from PyPi.utils.dataset import max_QA


class CollectQ(object):
    """
    This callback can be used to collect the values of the maximum action
    value in a given state at each call.
    """
    def __init__(self, approximator):
        """
        Constructor.

        Arguments
            approximator (object): the approximator to use;
        """
        self._approximator = approximator

        self._Qs = list()

    def __call__(self):
        if isinstance(self._approximator, Ensemble):
            qs = list()
            for m in self._approximator.models:
                qs.append(m.model._Q)
            self._Qs.append(deepcopy(np.mean(qs, 0)))
        else:
            self._Qs.append(deepcopy(self._approximator.model._Q))

    def get_values(self):
        return self._Qs


class CollectMaxQ(object):
    """
    This callback can be used to collect the values of the maximum action
    value in a given state at each call.
    """
    def __init__(self, approximator, state, action_values):
        """
        Constructor.

        Arguments
            approximator (object): the approximator to use;
            state (np.array): the state to consider;
            action_values (np.array): all the possible values of the action.
        """
        self._approximator = approximator
        self._state = state
        self._action_values = action_values

        self._max_Qs = list()

    def __call__(self):
        max_Q, _ = max_QA(self._state, False, self._approximator,
                          self._action_values)

        self._max_Qs.append(max_Q[0])

    def get_values(self):
        return self._max_Qs
