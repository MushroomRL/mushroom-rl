import numpy as np

from PyPi.algorithms.algorithm import Algorithm
from PyPi.utils.dataset import parse_dataset


class TD(Algorithm):
    """
    Implements functions to run TD algorithms.
    """
    def __init__(self, agent, mdp, **params):
        self.learning_rate = params.pop('learning_rate')

        super(TD, self).__init__(agent, mdp, **params)

    def fit(self, n_steps=1):
        """
        Updates Q function.

        # Arguments
            state (np.array): the current state
            action (np.array): the action performed in 'state'
            reward (np.array): the reward obtained applying 'action'
            next_state (np.array): the state reached applying 'action' in 'state'
            absorbing (np.array): flag indicating whether 'next_state' is absorbing
        """
        state, action, reward, next_state, absorbing, _ =\
            parse_dataset(np.array(self._dataset)[-1, :],
                          self.mdp.observation_space.dim,
                          self.mdp.action_space.dim)

        sa = np.concatenate((state, action), axis=1)
        q_current = self.agent.approximator.predict(sa)
        a_n = self._next_action(next_state, absorbing)
        sa_n = np.concatenate((next_state, a_n), axis=1)
        q_next = self.agent.approximator.predict(sa_n) * (1 - absorbing)

        q = q_current + self.learning_rate * (
            reward + self.gamma * q_next - q_current)

        self.agent.fit(sa, q, self.fit_params)

    def learn(self,
              n_iterations,
              how_many=1,
              n_fit_steps=1,
              iterate_over='samples'):
        super(TD, self).learn(n_iterations=n_iterations,
                              how_many=1,
                              n_fit_steps=1,
                              iterate_over='samples')


class QLearning(TD):
    """
    Q-Learning algorithm (Watkins, 1989).
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'Q-Learning'
        super(QLearning, self).__init__(agent, mdp, **params)

    def _next_action(self, next_state, absorbing):
        """
        Compute the action with the maximum action-value in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.
            absorbing (np.array): flag indicating whether 'next_state' is
                absorbing

        # Returns
            action with the maximum action_value in 'next_state'
        """
        return self.agent.draw_action(next_state, absorbing, True)


class SARSA(TD):
    """
    SARSA algorithm.
    """
    def __init__(self, agent, mdp, **params):
        self.__name__ = 'SARSA'
        super(SARSA, self).__init__(agent, mdp, **params)

    def _next_action(self, next_state, absorbing):
        """
        Compute the action returned by the policy in 'next_state'.

        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated
            absorbing (np.array): flag indicating whether 'next_state' is
                absorbing

        # Returns
            the action returned by the policy in 'next_state'.
        """
        return self.agent.draw_action(next_state, absorbing)
