from PyPi.algorithms import Algorithm
from PyPi.utils.dataset import select_samples


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

        super(DQN, self).__init__(agent, mdp, **params)

        self.target_network = agent.approximator

    def fit(self, _):
        """
        Single fit iteration on minibatch.

        # Arguments
            x (np.array): input dataset.
            y (np.array): target.
        """
        state, action, reward, next_states, absorbing, last =\
            select_samples(self._dataset,
                           self.mdp.observation_space.dim,
                           self.mdp.action_space.dim,
                           self.batch_size,
                           True)
        maxq, _ = self.agent.max_QA(next_states, absorbing, self.target_network)
        y = reward + self.gamma * maxq

        self.agent.train_on_batch(state, action, y, **self.fit_params)

    def updates(self):
        if self.iteration % self.target_update_frequency == 0:
            self.target_network.set_weights(
                self.agent.approximator.get_weights())

    def __str__(self):
        return self.__name__
