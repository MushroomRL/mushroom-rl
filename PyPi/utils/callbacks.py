from copy import deepcopy

import numpy as np
import tensorflow as tf

from PyPi.approximators.ensemble import Ensemble
from PyPi.utils.dataset import compute_scores, max_QA


class CollectDataset(object):
    def __init__(self):
        self._dataset = list()

    def __call__(self, *args):
        self._dataset += args[0]

    def get(self):
        return self._dataset


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

    def __call__(self, *args):
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
    def __init__(self, approximator, state):
        """
        Constructor.

        Arguments
            approximator (object): the approximator to use;
            state (np.array): the state to consider;
            action_values (np.array): all the possible values of the action.
        """
        self._approximator = approximator
        self._state = state

        self._max_Qs = list()

    def __call__(self, *args):
        max_Q, _ = max_QA(self._state, False, self._approximator)

        self._max_Qs.append(max_Q[0])

    def get_values(self):
        return self._max_Qs


class CollectSummary(object):
    def __init__(self, folder_name):
        self._summary_writer = tf.summary.FileWriter(folder_name)
        self._global_step = 0

    def __call__(self, dataset):
        score = compute_scores(dataset)

        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="min_reward",
                simple_value=score[0]),
            tf.Summary.Value(
                tag="max_reward",
                simple_value=score[1]),
            tf.Summary.Value(
                tag="average_reward",
                simple_value=score[2]),
            tf.Summary.Value(
                tag="games_completed",
                simple_value=score[3])]
        )
        self._summary_writer.add_summary(summary, self._global_step)

        self._global_step += 1
