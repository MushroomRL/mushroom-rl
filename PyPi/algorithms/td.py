import numpy as np

from PyPi.algorithms.algorithm import Algorithm
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter


class TD(Algorithm):
    """
    Implements functions to run TD algorithms.
    """
    def __init__(self, agent, mdp, **params):
        self.learning_rate = Parameter(params.pop('learning_rate'))

        super(TD, self).__init__(agent, mdp, **params)

    def fit(self, _):
        """
        Single fit step.
        """
        state, action, reward, next_state, absorbing, _ =\
            parse_dataset(np.array(self._dataset)[-1, :],
                          self.mdp.observation_space.dim,
                          self.mdp.action_space.dim)

        sa = np.concatenate((state, action), axis=1)
        q_current = self.agent.approximator.predict(sa)
        q_next = self._next_q(next_state, absorbing) * (1 - absorbing)

        q = q_current + self.learning_rate * (
            reward + self.gamma * q_next - q_current)

        self.agent.fit(sa, q, **self.fit_params)

    def callbacks(self):
        self.learning_rate()

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

    def _next_q(self, next_state, absorbing):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): whether 'next_state' is absorbing or not.

        # Returns
            Action with the maximum action_value in 'next_state'.
        """
        a_n = self.agent.draw_action(next_state, absorbing, True)
        sa_n = np.concatenate((next_state, a_n), axis=1)

        return self.agent.approximator.predict(sa_n)


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'DoubleQLearning'

        super(DoubleQLearning, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, absorbing):
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

    def _next_q(self, next_state, absorbing):
        pass


class SARSA(TD):
    """
    SARSA algorithm.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'SARSA'

        super(SARSA, self).__init__(agent, mdp, **params)

    def _next_q(self, next_state, absorbing):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): whether 'next_state' is absorbing or not.

        # Returns
            The action returned by the policy in 'next_state'.
        """
        a_n = self.agent.draw_action(next_state, absorbing)
        sa_n = np.concatenate((next_state, a_n), axis=1)

        return self.agent.approximator.predict(sa_n)
