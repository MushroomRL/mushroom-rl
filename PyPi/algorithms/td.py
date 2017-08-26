import numpy as np
from copy import deepcopy

from PyPi.algorithms.agent import Agent
from PyPi.utils.dataset import max_QA, parse_dataset


class TD(Agent):
    """
    Implements functions to run TD algorithms.
    """
    def __init__(self, approximator, policy, **params):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')

        super(TD, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        # Arguments
            dataset (list): the dataset to use.
        """
        assert n_iterations == 1 and len(dataset) == 1

        sample = dataset[0]
        sa = [np.array([sample[0]]), np.array([sample[1]])]

        q_current = self.approximator.predict(sa)
        q_next = self._next_q(np.array([sample[3]])) if not sample[4] else 0.

        q = q_current + self.learning_rate(sa) * (
            sample[2] + self.mdp_info['gamma'] * q_next - q_current)

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
        Arguments
            next_state (np.array): the state where next action has to be
                evaluated.

        # Returns
            Maximum action-value in 'next_state'.
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

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        # Arguments
            dataset (list): the dataset to use.
        """
        assert n_iterations == 1 and len(dataset) == 1

        sample = dataset[0]
        sa = [np.array([sample[0]]), np.array([sample[1]])]

        approximator_idx = 0 if np.random.uniform() < 0.5 else 1

        q_current = self.approximator[approximator_idx].predict(sa)
        q_next = self._next_q(
            np.array([sample[3]]), approximator_idx) if not sample[4] else 0.

        q = q_current + self.learning_rate[approximator_idx](sa) * (
            sample[2] + self.mdp_info['gamma'] * q_next - q_current)

        self.approximator[approximator_idx].fit(
            sa, q, **self.params['fit_params'])

    def _next_q(self, next_state, approximator_idx):
        """
        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated;
            approximator_idx (int): the index of the approximator to use
                to make the prediction.

        # Returns
            Action-value of the action whose value in 'next_state' is the
            maximum according to 'approximator[approximator]'.
        """
        _, a_n = max_QA(next_state, False,
                        self.approximator[approximator_idx],
                        self.mdp_info['action_space'].values)
        sa_n = [next_state, a_n]

        return self.approximator[1 - approximator_idx].predict(sa_n)


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'WeightedQLearning'

        self.sampling = params.pop('sampling', True)
        self.precision = params.pop('precision', 1000.)

        super(WeightedQLearning, self).__init__(approximator, policy, **params)

        self._n_updates = np.zeros(self.approximator.shape)
        self._sigma = np.ones(self.approximator.shape) * 1e10
        self._Q = np.zeros(self.approximator.shape)
        self._Q2 = np.zeros(self.approximator.shape)
        self._weights_var = np.zeros(self.approximator.shape)

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        # Arguments
            dataset (list): the dataset to use.
        """
        assert n_iterations == 1 and len(dataset) == 1

        sample = dataset[0]
        sa = [np.array([sample[0]]), np.array([sample[1]])]
        sa_idx = tuple(np.concatenate(
            (np.array([sample[0]]), np.array([sample[1]])),
            axis=1).astype(np.int).ravel())

        q_current = self.approximator.predict(sa)
        q_next = self._next_q(np.array([sample[3]])) if not sample[4] else 0.

        target = sample[2] + self.mdp_info['gamma'] * q_next

        alpha = self.learning_rate(sa)

        q = q_current + alpha * (target - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])

        self._n_updates[sa_idx] += 1

        self._Q[sa_idx] += (target - self._Q[sa_idx]) / self._n_updates[sa_idx]
        self._Q2[sa_idx] += (
            target ** 2. - self._Q2[sa_idx]) / self._n_updates[sa_idx]
        self._weights_var[sa_idx] = (1 - alpha) ** 2. * \
                                    self._weights_var[sa_idx] + alpha ** 2.

        if self._n_updates[sa_idx] > 1:
            var = self._n_updates[sa_idx] * (self._Q2[sa_idx] - self._Q[
                    sa_idx] ** 2.) / (self._n_updates[sa_idx] - 1.)
            var_estimator = var * self._weights_var[sa_idx]
            self._sigma[sa_idx] = np.sqrt(var_estimator)
            self._sigma[self._sigma < 1e-10] = 1e-10

    def _next_q(self, next_state):
        """
        # Arguments
            next_state (np.array): the state where next action has to be
                evaluated.

        # Returns
            the weighted estimator.
        """
        means = np.zeros((1, self.mdp_info['action_space'].n))
        sigmas = np.zeros(means.shape)
        actions = self.mdp_info['action_space'].values
        for i, a in enumerate(actions):
            sa_n = [next_state, np.array([a])]
            sa_n_idx = tuple(np.concatenate((next_state, np.array([a])),
                                            axis=1).astype(np.int).ravel())
            means[0, i] = self.approximator.predict(sa_n)
            sigmas[0, i] = self._sigma[sa_n_idx]

        if self.sampling:
            samples = np.random.normal(np.repeat(means, self.precision, 0),
                                       np.repeat(sigmas, self.precision, 0))
            max_idx = np.argmax(samples, axis=1)
            max_idx, max_count = np.unique(max_idx, return_counts=True)
            count = np.zeros(actions.shape[0])
            count[max_idx] = max_count

            w = count / self.precision
        else:
            raise NotImplementedError

        sa = [np.repeat(next_state, actions.shape[0], axis=0), actions]
        W = np.dot(w, self.approximator.predict(sa))

        return W


class SpeedyQLearning(TD):
    """
    Speedy Q-Learning algorithm.
    """
    def __init__(self, approximator, policy, **params):
        self.__name__ = 'SpeedyQLearning'

        self.old_q = deepcopy(approximator)

        super(SpeedyQLearning, self).__init__(approximator, policy, **params)

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        # Arguments
            dataset (list): the dataset to use.
        """
        assert n_iterations == 1 and len(dataset) == 1

        sample = dataset[0]
        sa = [np.array([sample[0]]), np.array([sample[1]])]

        # Save current q
        old_q = deepcopy(self.approximator)

        # Compute targets
        max_q_cur, _ = max_QA(np.array([sample[3]]), False, self.approximator,
                              self.mdp_info['action_space'].values)
        max_q_old, _ = max_QA(np.array([sample[3]]), False, self.old_q,
                              self.mdp_info['action_space'].values)

        target_cur = sample[2] + self.mdp_info['gamma'] * max_q_cur
        target_old = sample[2] + self.mdp_info['gamma'] * max_q_old

        # Update q
        alpha = self.learning_rate(sa)
        q_cur = self.approximator.predict(sa)
        q = q_cur + alpha * (target_old-q_cur) + (
            1.0 - alpha) * (target_cur - target_old)

        self.approximator.fit(sa, q, **self.params['fit_params'])

        # Update old q
        self.old_q = old_q


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
            the action_value of the action returned by the policy in
            'next_state'
        """
        self._next_action = self.draw_action(next_state)
        sa_n = [next_state, np.expand_dims(self._next_action, axis=0)]

        return self.approximator.predict(sa_n)
