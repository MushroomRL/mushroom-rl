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

    def max_QA(self, states, absorbing, target_approximator=None):
        """
        # Arguments
            state (np.array): the state where the agent is.
            absorbing (np.array): whether the state is absorbing or not.
            target_approximator (object, None): the model to use to predict
                the maximum Q-values.

        # Returns
            A np.array of maximum Q-values and a np.array of their corresponding
            action values.
        """
        n_states = states.shape[0]
        n_actions = self._discrete_actions.shape[0]
        action_dim = self._discrete_actions.shape[1]

        Q = np.zeros((n_states, n_actions))
        for action_idx in range(n_actions):
            actions = np.repeat(self._discrete_actions[action_idx],
                                n_states,
                                0).reshape(-1, 1)

            samples = (states, actions)

            if target_approximator is None:
                predictions = self.predict(samples)
            else:
                predictions = target_approximator.predict(samples)

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

    def draw_action(self, states, absorbing, force_max_action=False):
        """
        Compute an action according to the policy.

        # Arguments
            states (np.array): the state where the agent is.
            absorbing (np.array): whether the state is absorbing or not.
            force_max_action (bool): whether to select the best action or not.

        # Returns
            The selected action.
        """
        if not force_max_action:
            if self.policy():
                _, max_action = self.max_QA(states, absorbing)

                return max_action
            return np.array([self._discrete_actions[
                np.random.choice(range(self._discrete_actions.shape[0])), :]])
        else:
            _, max_action = self.max_QA(states, absorbing)

            return max_action

    def fit(self, x, y, **fit_params):
        """
        Fit the Q-function approximator.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).
            y (np.array): target.
            fit_params (dict): other parameters.
        """
        self.approximator.fit(x, y, **fit_params)

    def predict(self, x):
        """
        Predict using the Q-function approximator.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the Q-function approximator.
        """
        return self.approximator.predict(x)

    def train_on_batch(self, x, y, **fit_params):
        """
        Fit the Q-function approximator on one batch.

        # Arguments
            x (np.array): input dataset.
            y (np.array): target.
            fit_params (dict): other parameters.
        """
        self.approximator.train_on_batch(x, y, **fit_params)
