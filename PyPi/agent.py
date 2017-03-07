import numpy as np


class Agent(object):
    def __init__(self, approximator, policy, discrete_actions=None):
        self.approximator = approximator
        self.policy = policy
        if discrete_actions is None:
            self._discrete_actions = self.approximator.action_values
        else:
            self._discrete_actions = discrete_actions

    def max_QA(self, state, absorbing):
        n_states = state.shape[0]
        n_actions = self._discrete_actions.shape[0]
        action_dim = self._discrete_actions.shape[1]

        Q = np.zeros((n_states, n_actions))
        for action_idx in range(n_actions):
            actions = np.repeat(self._discrete_actions[action_idx],
                                n_states,
                                0)

            samples = np.column_stack((state, actions))

            predictions = self.predict(samples)

            Q[:, action_idx] = predictions * (1 - absorbing)

        if Q.shape[0] > 1:
            amax = np.argmax(Q, axis=1)
        else:
            q = Q[0]
            amax = [np.random.choice(np.argwhere(q == np.max(q)).ravel())]

        # store Q-value and action for each state
        r_q, r_a = np.zeros(n_states), np.zeros((n_states, action_dim))
        for idx in range(n_states):
            r_q[idx] = Q[idx, amax[idx]]
            r_a[idx] = self._discrete_actions[amax[idx]]

        return r_q, r_a

    def draw_action(self, state, absorbing, force_max_action=False):
        if not force_max_action:
            if not self.policy():
                _, max_action = self.max_QA(state, absorbing)

                return max_action
            return np.array([self._discrete_actions[
                np.random.choice(range(self._discrete_actions.shape[0])), :]])
        else:
            _, max_action = self.max_QA(state, absorbing)

            return max_action

    def fit(self, x, y):
        self.approximator.fit(x, y)

    def predict(self, x):
        return self.approximator.predict(x)
