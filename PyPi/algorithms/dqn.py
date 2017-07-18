from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset, state_action


class DQN(Agent):
    """
    Implements functions to run the DQN algorithm.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'DQN'

        alg_params = params['algorithm_params']
        self._target_approximator = alg_params.pop('target_approximator')
        self._initial_dataset_size = alg_params.pop('initial_dataset_size')
        self._target_update_frequency = alg_params.pop(
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
            state, action, reward, next_state, absorbing, _ = parse_dataset(
                dataset)

            sa = state_action(state, action)

            q_next = self._next_q(next_state) if not absorbing else 0.
            q = reward + self.mdp_info['gamma'] * q_next

            self.approximator.train_on_batch(sa, q, **self.params['fit_params'])

        self._n_updates += 1
        if not self._n_updates % self._target_update_frequency:
            self._target_approximator.set_weights(
                self.approximator.get_weights())

    def _next_q(self, next_state):
        """
        Arguments
            next_state (np.array): the state where next action has to be
                evaluated.

        # Returns
            Maximum action-value in 'next_state'.
        """
        max_q, _ = max_QA(next_state, False, self._target_approximator,
                          self.mdp_info['action_space'].values)

    def __str__(self):
        return self.__name__
