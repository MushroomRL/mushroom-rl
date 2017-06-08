import numpy as np
from copy import deepcopy

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset
from PyPi.utils import spaces


class TD(Agent):
    """
    Implements functions to run TD algorithms.
    """
    def __init__(self, approximator, policy, **params):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')

        super(TD, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1

        state, action, reward, next_state, absorbing, _ = parse_dataset(
            [dataset[-1]])

        sa = [state, action]
        sa_idx = np.concatenate((
            self.mdp_info['observation_space'].get_idx(state),
            self.mdp_info['action_space'].get_idx(action)),
            axis=1)
        q_current = self.approximator.predict(sa)
        q_next = self._next_q(next_state) if not absorbing else 0.

        q = q_current + self.learning_rate(sa_idx) * (
            reward + self.mdp_info['gamma'] * q_next - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])

    def __str__(self):
        return self.__name__


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'QLearning'

        super(QLearning, self).__init__(approximator, policy, **params)

    def _next_q(self, next_state):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.

        # Returns
            Action with the maximum action_value in 'next_state'.
        """
        max_q, _ = max_QA(next_state, False, self.approximator,
                          self.mdp_info['action_space'].values)

        return max_q


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'DoubleQLearning'

        super(DoubleQLearning, self).__init__(approximator, policy, **params)

        self.learning_rate = [deepcopy(self.learning_rate),
                              deepcopy(self.learning_rate)]

        assert self.approximator.n_models == 2, 'The regressor ensemble must' \
                                                ' have exactly 2 models.'

    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1

        state, action, reward, next_state, absorbing, _ = parse_dataset(
            [dataset[-1]])

        sa = [state, action]
        sa_idx = np.concatenate((
            self.mdp_info['observation_space'].get_idx(state),
            self.mdp_info['action_space'].get_idx(action)),
            axis=1)

        approximator_idx = 0
        if np.random.uniform() < 0.5:
            approximator_idx = 1

        q_current = self.approximator[approximator_idx].predict(sa)
        q_next = self._next_q(
            next_state, approximator_idx) if not absorbing else 0.

        q = q_current + self.learning_rate[approximator_idx](sa_idx) * (
            reward + self.mdp_info['gamma'] * q_next - q_current)

        self.approximator[approximator_idx].fit(
            sa, q, **self.params['fit_params'])

    def _next_q(self, next_state, approximator_idx):
        _, a_n_idx = max_QA(next_state, False,
                            self.approximator[approximator_idx],
                            self.mdp_info['action_space'].values)
        a_n_value = self.mdp_info['action_space'].get_value(a_n_idx)
        sa_n = [next_state, a_n_value]

        return self.approximator[1 - approximator_idx].predict(sa_n)


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'WeightedQLearning'

        self.sampling = params.pop('sampling', False)
        self.precision = params.pop('precision', 1000)

        super(WeightedQLearning, self).__init__(approximator, policy, **params)

    def _next_q(self, next_state):
        pass


class SARSA(TD):
    """
    SARSA algorithm.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'SARSA'

        super(SARSA, self).__init__(approximator, policy, **params)

    def _next_q(self, next_state):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.

        # Returns
            ...
        """
        self._next_action = self.draw_action(next_state)
        action_value = self.mdp_info[
            'action_space'].get_value(self._next_action)
        sa_n = [next_state, action_value]

        return self.approximator.predict(sa_n)
