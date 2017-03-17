import numpy as np

from PyPi.algorithms import Algorithm
from PyPi.utils.dataset import parse_dataset


class BatchTD(Algorithm):
    """
    Implements functions to run Batch algorithms.
    """
    def __init__(self, agent, mdp, **params):
        super(BatchTD, self).__init__(agent, mdp, **params)


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm (Ernst, 2005).
    """
    def __init__(self, agent, mdp, **params):
        super(FQI, self).__init__(agent, mdp, **params)

    def fit(self, n_iterations):
        """
        Fit loop.

        # Arguments
            n_iterations (int > 0): number of iterations
        """
        target = None
        for i in range(n_iterations):
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

    def learn(self,
              n_iterations=1,
              how_many=100,
              n_fit_steps=20,
              iterate_over='episodes',
              render=False):
        super(FQI, self).learn(n_iterations=n_iterations,
                               how_many=how_many,
                               n_fit_steps=n_fit_steps,
                               iterate_over=iterate_over,
                               render=render)


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm (D'Eramo et. al., 2017).
    """
    def __init__(self, agent, mdp, **params):
        super(DoubleFQI, self).__init__(agent, mdp, **params)

    def partial_fit(self, x, y):
        pass


class WeightedFQI(FQI):
    """
    Weighted Fitted Q-Iteration algorithm (D'Eramo et. al., 2017).
    """
    def __init__(self, agent, mdp, **params):
        super(WeightedFQI, self).__init__(agent, mdp, **params)

    def partial_fit(self, x, y):
        pass
