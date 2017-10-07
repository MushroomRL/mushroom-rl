import numpy as np
from copy import deepcopy

from mushroom.algorithms.agent import Agent
from mushroom.approximators import EnsembleTable
from mushroom.utils.table import Table


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
        s = sample[0]
        a = sample[1]
        r = sample[2]
        ss = sample[3]
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
    def __init__(self, shape, policy, gamma, **params):
        self.__name__ = 'QLearning'

        self.Q = Table(shape)

        super(QLearning, self).__init__(self.Q, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        q_current = self.Q[s, a]

        q_next = np.max(self.Q[ss, :]) if not ab else 0.

        self.Q[s, a] = q_current + self.learning_rate(s, a) * (
             r + self._gamma * q_next - q_current)


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.

    """
    def __init__(self, shape, policy, gamma, **params):
        self.__name__ = 'DoubleQLearning'

        self.Q = EnsembleTable(2, shape)

        super(DoubleQLearning, self).__init__(self.Q, policy, gamma, **params)

        self.learning_rate = [deepcopy(self.learning_rate),
                              deepcopy(self.learning_rate)]

        assert len(self.Q) == 2, 'The regressor ensemble must' \
                                 ' have exactly 2 models.'

    def _update(self, s, a, r, ss, ab):
        approximator_idx = 0 if np.random.uniform() < .5 else 1

        q_current = self.Q[approximator_idx][s, a]

        if not ab:
            q_ss = self.Q[approximator_idx][ss, :]
            max_q = np.max(q_ss)
            a_n = np.array(
                [np.random.choice(np.argwhere(q_ss == max_q).ravel())])
            q_next = self.Q[1 - approximator_idx][ss, a_n]
        else:
            q_next = 0.

        q = q_current + self.learning_rate[approximator_idx](s, a) * (
            r + self._gamma * q_next - q_current)

        self.Q[approximator_idx][s, a] = q


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.

    """
    def __init__(self, shape, policy, gamma, **params):
        self.__name__ = 'WeightedQLearning'

        self.Q = Table(shape)
        self._sampling = params.pop('sampling', True)
        self._precision = params.pop('precision', 1000)

        super(WeightedQLearning, self).__init__(self.Q, policy, gamma, **params)

        self._n_updates = Table(shape)
        self._sigma = Table(shape, initial_value=1e10)
        self._Q = Table(shape)
        self._Q2 = Table(shape)
        self._weights_var = Table(shape)

    def _update(self, s, a, r, ss, ab):
        q_current = self.Q[s, a]
        q_next = self._next_q(ss) if not ab else 0.

        target = r + self._gamma * q_next

        alpha = self.learning_rate(s, a)

        self.Q[s, a] = q_current + alpha * (target - q_current)

        self._n_updates[s, a] += 1

        self._Q[s, a] += (target - self._Q[s, a]) / self._n_updates[s, a]
        self._Q2[s, a] += (
            target ** 2. - self._Q2[s, a]) / self._n_updates[s, a]
        self._weights_var[s, a] = (1 - alpha) ** 2. * self._weights_var[
            s, a] + alpha ** 2.

        if self._n_updates[s, a] > 1:
            var = self._n_updates[s, a] * (self._Q2[s, a] - self._Q[
                    s, a] ** 2.) / (self._n_updates[s, a] - 1.)
            var_estimator = var * self._weights_var[s, a]
            var_estimator = var_estimator if var_estimator >= 1e-10 else 1e-10
            self._sigma[s, a] = np.sqrt(var_estimator)

    def _next_q(self, next_state):
        """
        Args:
            next_state (np.array): the state where next action has to be
                evaluated.

        Returns:
            The weighted estimator value in 'next_state'.

        """
        means = self.Q[next_state, :]
        sigmas = np.zeros(self.Q.shape[-1])

        for a in xrange(sigmas.size):
            sigmas[a] = self._sigma[next_state, np.array([a])]

        if self._sampling:
            samples = np.random.normal(np.repeat([means], self._precision, 0),
                                       np.repeat([sigmas], self._precision, 0))
            max_idx = np.argmax(samples, axis=1)
            max_idx, max_count = np.unique(max_idx, return_counts=True)
            count = np.zeros(means.size)
            count[max_idx] = max_count

            w = count / self._precision
        else:
            raise NotImplementedError

        return np.dot(w, means)


class SpeedyQLearning(TD):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.

    """
    def __init__(self, shape, policy, gamma, **params):
        self.__name__ = 'SpeedyQLearning'

        self.Q = Table(shape)
        self.old_q = deepcopy(self.Q)

        super(SpeedyQLearning, self).__init__(self.Q, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        old_q = deepcopy(self.Q)

        max_q_cur = np.max(self.Q[ss, :]) if not ab else 0.
        max_q_old = np.max(self.old_q[ss, :]) if not ab else 0.

        target_cur = r + self._gamma * max_q_cur
        target_old = r + self._gamma * max_q_old

        alpha = self.learning_rate(s, a)
        q_cur = self.Q[s, a]
        self.Q[s, a] = q_cur + alpha * (target_old-q_cur) + (
            1. - alpha) * (target_cur - target_old)

        self.old_q = old_q


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, shape, policy, gamma, **params):
        self.__name__ = 'SARSA'

        self.Q = Table(shape)
        super(SARSA, self).__init__(self.Q, policy, gamma, **params)

    def _update(self, s, a, r, ss, ab):
        q_current = self.Q[s, a]

        self._next_action = self.draw_action(ss)
        q_next = self.Q[ss, self._next_action] if not ab else 0.

        self.Q[s, a] = q_current + self.learning_rate(s, a) * (
             r + self._gamma * q_next - q_current)
