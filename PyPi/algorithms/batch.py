import numpy as np

from PyPi.algorithms import Algorithm
from PyPi.utils.dataset import parse_dataset


class Batch(Algorithm):
    def __init__(self, agent, mdp, **params):
        super(Batch, self).__init__(agent, mdp, **params)


class FQI(Batch):
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'FQI'
        super(FQI, self).__init__(agent, mdp, **params)

    def fit(self, n_steps):
        target = None
        for i in range(n_steps):
            target = self.partial_fit(self._dataset, target)

    def partial_fit(self, x, y):
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
        self.agent.fit(sa, y)

        return y

    def learn(self,
              n_iterations=1,
              how_many=100,
              n_fit_steps=20,
              iterate_over='episodes'):
        super(FQI, self).learn(n_iterations=n_iterations,
                               how_many=how_many,
                               n_fit_steps=n_fit_steps,
                               iterate_over='episodes')
