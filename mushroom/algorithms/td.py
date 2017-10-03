import numpy as np
from copy import deepcopy

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import max_QA


class TD(Agent):
    """
    Implements functions to run TD algorithms.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')

        super(TD, self).__init__(approximator, policy, gamma, **params)

    def fit(self, dataset, n_iterations=1):
        """
        Single fit step.

        Args:
            dataset (list):  a two elements list with the state and the action;
            n_iterations (int, 1): number of fit steps of the approximator.

        """
        assert n_iterations == 1 and len(dataset) == 1

        s, a, r, ss, ab = self._parse(dataset)
        self._update(s, a, r, ss, ab)

    @staticmethod
    def _parse(dataset):
        sample = dataset[0]
        s = np.array([sample[0]])
        a = np.array([sample[1]])
        r = sample[2]
        ss = np.array([sample[3]])
        ab = sample[4]
        
        return s, a, r, ss, ab

    def _update(self, s, a, r, ss, ab):
        pass

    def __str__(self):
        return self.__name__


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'QLearning'

        super(QLearning, self).__init__(approximator, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        sa = [s, a]
        q_current = self.approximator.predict(sa)

        if not ab:
            q_next, _ = max_QA(ss, False, self.approximator)
        else:
            q_next = 0

        q = q_current + self.learning_rate(sa) * (
             r + self._gamma * q_next - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'DoubleQLearning'

        super(DoubleQLearning, self).__init__(approximator, policy, gamma, **params)

        self.learning_rate = [deepcopy(self.learning_rate),
                              deepcopy(self.learning_rate)]

        assert len(self.approximator) == 2, 'The regressor ensemble must' \
                                            ' have exactly 2 models.'

    def _update(self, s, a, r, ss, ab):
        sa = [s, a]

        approximator_idx = 0 if np.random.uniform() < 0.5 else 1

        q_current = self.approximator[approximator_idx].predict(sa)

        if not ab:
            _, a_n = max_QA(ss, False, self.approximator[approximator_idx])
            a_n = np.array([[np.random.choice(a_n.ravel())]])
            sa_n = [ss, a_n]

            q_next = self.approximator[1 - approximator_idx].predict(sa_n)
        else:
            q_next = 0.

        q = q_current + self.learning_rate[approximator_idx](sa) * (
            r + self._gamma * q_next - q_current)

        self.approximator[approximator_idx].fit(
            sa, q, **self.params['fit_params'])


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'WeightedQLearning'

        self._sampling = params.pop('sampling', True)
        self._precision = params.pop('precision', 1000.)

        super(WeightedQLearning, self).__init__(approximator, policy, gamma, **params)

        self._n_updates = np.zeros(self.approximator.shape)
        self._sigma = np.ones(self.approximator.shape) * 1e10
        self._Q = np.zeros(self.approximator.shape)
        self._Q2 = np.zeros(self.approximator.shape)
        self._weights_var = np.zeros(self.approximator.shape)

    def _update(self, s, a, r, ss, ab):
        sa = [s, a]
        sa_idx = tuple(np.concatenate((s, a), axis=1).astype(np.int).ravel())

        q_current = self.approximator.predict(sa)
        q_next = self._next_q(ss) if not ab else 0.

        target = r + self._gamma * q_next

        alpha = self.learning_rate(sa)

        q = q_current + alpha * (target - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])

        self._n_updates[sa_idx] += 1

        self._Q[sa_idx] += (target - self._Q[sa_idx]) / self._n_updates[sa_idx]
        self._Q2[sa_idx] += (
            target ** 2. - self._Q2[sa_idx]) / self._n_updates[sa_idx]
        self._weights_var[sa_idx] = (1 - alpha) ** 2. * self._weights_var[
            sa_idx] + alpha ** 2.

        if self._n_updates[sa_idx] > 1:
            var = self._n_updates[sa_idx] * (self._Q2[sa_idx] - self._Q[
                    sa_idx] ** 2.) / (self._n_updates[sa_idx] - 1.)
            var_estimator = var * self._weights_var[sa_idx]
            self._sigma[sa_idx] = np.sqrt(var_estimator)
            self._sigma[self._sigma < 1e-10] = 1e-10

    def _next_q(self, next_state):
        """
        Args:
            next_state (np.array): the state where next action has to be
                evaluated.

        Returns:
            The weighted estimator value in 'next_state'.

        """
        means = self.approximator.predict_all(next_state)

        sigmas = np.zeros((1, self.approximator.Q.shape[-1]))
        for a in xrange(sigmas.size):
            sa_n_idx = tuple(np.concatenate((next_state, np.array([[a]])),
                                            axis=1).astype(np.int).ravel())
            sigmas[0, a] = self._sigma[sa_n_idx]

        if self._sampling:
            samples = np.random.normal(np.repeat(means, self._precision, 0),
                                       np.repeat(sigmas, self._precision, 0))
            max_idx = np.argmax(samples, axis=1)
            max_idx, max_count = np.unique(max_idx, return_counts=True)
            count = np.zeros(means.size)
            count[max_idx] = max_count

            w = count / self._precision
        else:
            raise NotImplementedError

        return np.dot(w, means.T)[0]


class SpeedyQLearning(TD):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'SpeedyQLearning'

        self.old_q = deepcopy(approximator)

        super(SpeedyQLearning, self).__init__(approximator, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        sa = [s, a]

        old_q = deepcopy(self.approximator)

        max_q_cur, _ = max_QA(ss, False, self.approximator)
        max_q_old, _ = max_QA(ss, False, self.old_q)

        target_cur = r + self._gamma * max_q_cur
        target_old = r + self._gamma * max_q_old

        alpha = self.learning_rate(sa)
        q_cur = self.approximator.predict(sa)
        q = q_cur + alpha * (target_old-q_cur) + (
            1.0 - alpha) * (target_cur - target_old)

        self.approximator.fit(sa, q, **self.params['fit_params'])

        self.old_q = old_q


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'SARSA'

        super(SARSA, self).__init__(approximator, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        sa = [s, a]
        q_current = self.approximator.predict(sa)

        self._next_action = self.draw_action(ss)
        sa_n = [ss, np.expand_dims(self._next_action, axis=0)]

        q_next = self.approximator.predict(sa_n) if not ab else 0

        q = q_current + self.learning_rate(sa) * (
             r + self._gamma * q_next - q_current)

        self.approximator.fit(sa, q, **self.params['fit_params'])
