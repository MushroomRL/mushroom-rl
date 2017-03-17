import numpy as np


class Agent(object):
    """
    This class implements the functions to evaluate the Q-function
    of the agent and drawing actions.
    """
    def __init__(self, approximator, policy, discrete_actions):
        """
        Constructor.

        # Arguments
            approximator (object): the approximator of the Q function.
            policy (object): the policy to use.
            discrete_actions (np.array): the array containing the discretized
                                         action values.
        """
        self.approximator = approximator
        self.policy = policy
        self._discrete_actions = discrete_actions

    def max_QA(self, state, absorbing):
        """
        # Arguments
            state (np.array): the state where the agent is.
            absorbing (np.array): whether the state is absorbing or not

        # Returns
            an array of maximum Q-values and an array of their corresponding
            actions values.
        """
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
        """
        Compute an action according to the policy.

        # Arguments
            state (np.array): the state where the agent is.
            absorbing (np.array): whether the state is absorbing or not.
            force_max_action (bool): whether to select the best action or not.

        # Returns
            the selected action.
        """
        if not force_max_action:
            if self.policy():
                _, max_action = self.max_QA(state, absorbing)

                return max_action
            return np.array([self._discrete_actions[
                np.random.choice(range(self._discrete_actions.shape[0])), :]])
        else:
            _, max_action = self.max_QA(state, absorbing)

            return max_action

    def fit(self, x, y, **fit_params):
        self.approximator.fit(x, y, **fit_params)

    def predict(self, x):
        return self.approximator.predict(x)
