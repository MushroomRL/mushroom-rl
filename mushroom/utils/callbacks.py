from copy import deepcopy

import numpy as np
import tensorflow as tf

from mushroom.approximators.ensemble import Ensemble
from mushroom.utils.dataset import compute_scores, max_QA


class CollectDataset:
    """
    This callback can be used to collect the samples during the run of the
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

        self._Qs = list()

    def __call__(self, **kwargs):
        if isinstance(self._approximator, Ensemble):
            qs = list()
            for m in self._approximator.models:
                qs.append(m.model.Q)
            self._Qs.append(deepcopy(np.mean(qs, 0)))
        else:
            self._Qs.append(deepcopy(self._approximator.model.Q))

    def get_values(self):
        return self._Qs


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

        self._max_Qs = list()

    def __call__(self, **kwargs):
        max_Q, _ = max_QA(self._state, False, self._approximator)

        self._max_Qs.append(max_Q[0])

    def get_values(self):
        return self._max_Qs


class CollectSummary:
    """
    This callback can be used to collect the tensorflow summary to be plotted
    in tensorboard.

    """
    def __init__(self, folder_name):
        self._summary_writer = tf.summary.FileWriter(folder_name)
        self._global_step = 0

    def __call__(self, **kwargs):
        score = compute_scores(kwargs['dataset'])

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
