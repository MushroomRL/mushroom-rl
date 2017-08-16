import numpy as np

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, select_samples


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et. al.. 2015.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'DQN'

        alg_params = params['algorithm_params']
        self._batch_size = alg_params.get('batch_size')
        self._clip_reward = alg_params.get('clip_reward', True)
        self._target_approximator = alg_params.get('target_approximator')
        self._initial_dataset_size = alg_params.get('initial_dataset_size')
        self._target_update_frequency = alg_params.get(
            'target_update_frequency')
        self._n_updates = 0

        super(DQN, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        # Arguments
            dataset (list): the dataset to use.
        """
        assert n_iterations == 1

        if len(dataset) >= self._initial_dataset_size:
            state, action, reward, next_state, absorbing, _ = select_samples(
                dataset=dataset, n_samples=self._batch_size, parse=True)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            sa = [state, action]

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info['gamma'] * q_next

            self.approximator.train_on_batch(sa, q, **self.params['fit_params'])

            if self._n_updates > 0 and self._n_updates %\
                    self._target_update_frequency == 0:
                self._target_approximator.model.set_weights(
                    self.approximator.model.get_weights())

            self._n_updates += 1

    def _next_q(self, next_state, absorbing):
        """
        Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): the absorbing flag for the states in
                'next_state'.

        # Returns
            Maximum action-value in 'next_state'.
        """
        max_q, _ = max_QA(next_state, absorbing, self.approximator,
                          self.mdp_info['action_space'].values)

        return max_q

    def __str__(self):
        return self.__name__


class DoubleDQN(DQN):
    """
    Implements functions to run the Double DQN algorithm.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'DoubleDQN'

        super(DoubleDQN, self).__init__(approximator, policy, **params)

    def _next_q(self, next_state, absorbing):
        """
        Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): the absorbing flag for the states in
                'next_state'.

        # Returns
            Maximum action-value in 'next_state'.
        """
        q = self._approximator.predict(next_state)
        max_a = np.argmax(q, axis=1).reshape(-1, 1)
        sa_n = [next_state, max_a]

        return self._target_approximator.predict(sa_n)
