import numpy as np

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter


class TD(Agent):
    """
    Implements functions to run TD algorithms.
    """
    def __init__(self, agent, mdp, **params):
        self.learning_rate = Parameter(
            params['algorithm_params'].pop('learning_rate'))

        super(TD, self).__init__(agent, mdp, **params)

    def fit(self, dataset, n_fit_iterations=1):
        """
        Single fit step.
        """
        assert n_fit_iterations == 1

        state, action, reward, next_state, absorbing, _ =\
            parse_dataset(np.array(dataset)[-1, :],
                          self.mdp_info['observation_space'].dim,
                          self.mdp_info['action_space'].dim)

        sa = (state, action)
        q_current = self.approximator.predict(sa)
        q_next = self._next_q(
            next_state, self.approximator) if not absorbing else 0

        q = q_current + self.learning_rate() * (
            reward + self.mdp_info['gamma'] * q_next - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])

    def updates(self):
        self.learning_rate.update()

    def __str__(self):
        return self.__name__


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'QLearning'

        super(QLearning, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, approximator, greedy_action=True):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): whether 'next_state' is absorbing or not.

        # Returns
            Action with the maximum action_value in 'next_state'.
        """
        a_n = self.draw_action(next_state, approximator, greedy_action)
        sa_n = (next_state, a_n)

        return self.approximator.predict(sa_n)


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'DoubleQLearning'

        super(DoubleQLearning, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, approximator, greedy_action):
        pass


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'WeightedQLearning'

        self.exact = params.pop('exact', True)

        super(WeightedQLearning, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, approximator, greedy_action):
        pass


class SARSA(TD):
    """
    SARSA algorithm.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'SARSA'

        super(SARSA, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, approximator, greedy_action=False):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): whether 'next_state' is absorbing or not.

        # Returns
            The action returned by the policy in 'next_state'.
        """
        a_n = self.draw_action(next_state, approximator, greedy_action)
        sa_n = [next_state, a_n]

        return self.approximator.predict(sa_n)
