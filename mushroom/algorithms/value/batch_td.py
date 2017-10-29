import numpy as np
from tqdm import tqdm

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import parse_dataset


class BatchTD(Agent):
    """
    Implement functions to run batch algorithms.

    """
    def __init__(self, approximator, policy, gamma, params):
        """
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.

        """
        self._quiet = params['algorithm_params'].get('quiet', False)

        policy.set_q(approximator)
        self.approximator = approximator

        super(BatchTD, self).__init__(policy, gamma, params)

    def __str__(self):
        return self.__name__


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, gamma, params):
        self.__name__ = 'FQI'

        super(FQI, self).__init__(approximator, policy, gamma, params)

        # "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.
        self._boosted = params['algorithm_params'].get('boosted', False)
        if self._boosted:
            self._prediction = 0.
            self._next_q = 0.
            self._idx = 0

    def fit(self, dataset, n_iterations, target=None):
        """
        Fit loop.

        Args:
            dataset (list): the dataset;
            n_iterations (int): number of FQI iterations;
            target (np.array, None): initial target of FQI.

        Returns:
            Last target computed.

        """
        if self._boosted:
            if target is None:
                self._prediction = 0.
                self._next_q = 0.
                self._idx = 0
            for _ in tqdm(xrange(n_iterations), dynamic_ncols=True,
                          disable=self._quiet, leave=False):
                target = self._partial_fit_boosted(dataset, target)
        else:
            for _ in tqdm(xrange(n_iterations), dynamic_ncols=True,
                          disable=self._quiet, leave=False):
                target = self._partial_fit(dataset, target)

        return target

    def _partial_fit(self, x, y):
        """
        Single fit iteration.

        Args:
            x (list): the dataset;
            y (np.array): targets.

        Returns:
            Last target computed.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if y is None:
            target = reward
        else:
            q = self.approximator.predict(next_state)
            if np.any(absorbing):
                q *= 1 - absorbing.reshape(-1, 1)

            max_q = np.max(q, axis=1)
            target = reward + self._gamma * max_q

        self.approximator.fit(state, action, target,
                              **self.params['fit_params'])

        return target

    def _partial_fit_boosted(self, x, y):
        """
        Single fit iteration for boosted FQI.

        Args:
            x (list): the dataset;
            y (np.array): targets.

        Returns:
            Last target computed.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if y is None:
            target = reward
        else:
            self._next_q += self.approximator.predict(next_state,
                                                      idx=self._idx - 1)
            if np.any(absorbing):
                self._next_q *= 1 - absorbing.reshape(-1, 1)

            max_q = np.max(self._next_q, axis=1)
            target = reward + self._gamma * max_q

        target = target - self._prediction
        self._prediction += target

        self.approximator.fit(state, action, target, idx=self._idx)

        self._idx += 1

        return target


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et al.. 2017.

    """
    def __init__(self, approximator, policy, gamma, params):
        self.__name__ = 'DoubleFQI'

        super(DoubleFQI, self).__init__(approximator, policy, gamma, params)

    def _partial_fit(self, x, y, **fit_params):
        pass


class WeightedFQI(FQI):
    """
    Weighted Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et al.. 2017.

    """
    def __init__(self, approximator, policy, gamma, params):
        self.__name__ = 'WeightedFQI'

        super(WeightedFQI, self).__init__(approximator, policy, gamma, params)

    def _partial_fit(self, x, y, **fit_params):
        pass
