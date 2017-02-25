import numpy as np

from PyPi.algorithms.algorithm import Algorithm


class TD(Algorithm):
    def __init__(self, agent, mdp, logger, **params):
        self.learning_rate = params.pop('learning_rate')

        super(TD, self).__init__(agent, mdp, logger, **params)

    def step(self, state, action, reward, next_state, absorbing):
        sa = np.concatenate((state, action), axis=1)
        q_current = self.agent.approximator.predict(sa)
        a_n = self._next_action(next_state, absorbing)
        sa_n = np.concatenate((next_state, a_n), axis=1)
        q_next = self.agent.approximator.predict(sa_n) * (1 - absorbing)

        q = q_current + self.learning_rate * (
            reward + self.gamma * q_next - q_current)

        self.agent.approximator.fit(sa, q)


class QLearning(TD):
    def __init__(self, agent, mdp, logger, **params):
        self.__name__ = 'Q-Learning'
        super(QLearning, self).__init__(agent, mdp, logger, **params)

    def _next_action(self, next_state, absorbing):
        return self.agent.draw_action(next_state, absorbing, True)


class SARSA(TD):
    def __init__(self, agent, mdp, logger, **params):
        self.__name__ = 'SARSA'
        super(SARSA, self).__init__(agent, mdp, logger, **params)

    def _next_action(self, next_state, absorbing):
        return self.agent.draw_action(next_state, absorbing)
