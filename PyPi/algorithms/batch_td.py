import numpy as np

from PyPi.algorithms import Algorithm
from PyPi.utils.dataset import parse_dataset


class BatchTD(Algorithm):
    """
    Implements functions to run Batch algorithms.
    """
    def __init__(self, agent, mdp, **params):
        super(BatchTD, self).__init__(agent, mdp, **params)

    def updates(self):
        pass

    def __str__(self):
        return self.__name__


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et.al.. 2005.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'FQI'

        super(FQI, self).__init__(agent, mdp, **params)

    def fit(self, n_iterations):
        """
        Fit loop.

        # Arguments
            n_iterations (int > 0): number of iterations
        """
        target = None
        for i in range(n_iterations):
            self.logger.info('Iteration: %d' % (i + 1))
            target = self.partial_fit(self._dataset, target)

    def partial_fit(self, x, y):
        """
        Single fit iteration.

        # Arguments
            x (np.array): input dataset.
            y (np.array): target.
        """
        state, action, reward, next_states, absorbing, last =\
            parse_dataset(x,
                          self.mdp.observation_space.dim,
                          self.mdp.action_space.dim)
        if y is None:
            y = reward
        else:
            maxq, _ = self.agent.max_QA(next_states, absorbing)
            y = reward + self.gamma * maxq

        sa = np.concatenate((state, action), axis=1)

        self.agent.fit(sa, y, **self.fit_params)

        return y


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et. al.. 2017.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'DoubleFQI'

        super(DoubleFQI, self).__init__(agent, mdp, **params)

    def partial_fit(self, x, y):
        pass


class WeightedFQI(FQI):
    """
    Weighted Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et. al.. 2017.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'WeightedFQI'

        super(WeightedFQI, self).__init__(agent, mdp, **params)

    def partial_fit(self, x, y):
        pass
