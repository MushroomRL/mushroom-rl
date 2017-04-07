import numpy as np

from PyPi.algorithms import Algorithm
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter


class DQN(Algorithm):
    """
    Deep Q-Network algorithm.
    "Human-Level Control through Deep Reinforcement Learning".
    Mnih et. al.. 2015.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'DQN'

        self.batch_size = params.pop('batch_size', 32)
        self.target_update_frequency = params.pop('target_update_frequency',
                                                  1e4)
        self.learning_rate = Parameter(params.pop('learning_rate'))

        super(DQN, self).__init__(agent, mdp, **params)

        self.target_network = agent.approximator.clone()

    def fit(self, _):
        """
        Single fit iteration on minibatch.

        # Arguments
            x (np.array): input dataset.
            y (np.array): target.
        """
        idxs = np.random.randint(self._dataset.shape[0], size=self.batch_size)
        x = self._dataset[idxs, ...]

        state, action, reward, next_states, absorbing, last =\
            parse_dataset(x,
                          self.mdp.observation_space.dim,
                          self.mdp.action_space.dim)
        maxq, _ = self.agent.max_QA(next_states, absorbing, self.target_network)
        y = reward + self.gamma * maxq

        sa = [state, action]

        self.agent.train_on_batch(sa, y, **self.fit_params)

    def callbacks(self):
        self.learning_rate()
        if self.iteration % self.target_update_frequency == 0:
            self.target_network.set_weights(
                self.agent.approximator.get_weights())
