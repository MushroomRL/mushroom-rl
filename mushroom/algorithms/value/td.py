import numpy as np
from copy import deepcopy

from mushroom.algorithms.agent import Agent
from mushroom.approximators import EnsembleTable
from mushroom.utils.table import Table


class TD(Agent):
    """
    Implements functions to run TD algorithms.

    """
    def __init__(self, approximator, policy, gamma, params):
        self.alpha = params['algorithm_params']['learning_rate']

        policy.set_q(approximator)
        self.approximator = approximator

        super(TD, self).__init__(policy, gamma, params)

    def fit(self, dataset, n_iterations=1):
        assert n_iterations == 1 and len(dataset) == 1

        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state, absorbing)

    @staticmethod
    def _parse(dataset):
        sample = dataset[0]
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        
        return state, action, reward, next_state, absorbing

    def _update(self, state, action, reward, next_state, absorbing):
        """
        Update the Q-table.

        Args:
            state (np.array): state;
            action (np.array): action;
            reward (np.array): reward;
            next_state (np.array): next state;
            absorbing (np.array): absorbing flag.

        """
        pass

    def __str__(self):
        return self.__name__


class QLearning(TD):
    """
    Q-Learning algorithm.
    "Learning from Delayed Rewards". Watkins C.J.C.H.. 1989.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'QLearning'

        self.Q = Table(shape)

        super(QLearning, self).__init__(self.Q, policy, gamma, params)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.

        self.Q[state, action] = q_current + self.alpha(
            state, action) * (reward + self._gamma * q_next - q_current)


class DoubleQLearning(TD):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". van Hasselt H.. 2010.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'DoubleQLearning'

        self.Q = EnsembleTable(2, shape)

        super(DoubleQLearning, self).__init__(self.Q, policy, gamma, params)

        self.alpha = [deepcopy(self.alpha), deepcopy(self.alpha)]

        assert len(self.Q) == 2, 'The regressor ensemble must' \
                                 ' have exactly 2 models.'

    def _update(self, state, action, reward, next_state, absorbing):
        approximator_idx = 0 if np.random.uniform() < .5 else 1

        q_current = self.Q[approximator_idx][state, action]

        if not absorbing:
            q_ss = self.Q[approximator_idx][next_state, :]
            max_q = np.max(q_ss)
            a_n = np.array(
                [np.random.choice(np.argwhere(q_ss == max_q).ravel())])
            q_next = self.Q[1 - approximator_idx][next_state, a_n]
        else:
            q_next = 0.

        q = q_current + self.alpha[approximator_idx](state, action) * (
            reward + self._gamma * q_next - q_current)

        self.Q[approximator_idx][state, action] = q


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'WeightedQLearning'

        self.Q = Table(shape)
        self._sampling = params.pop('sampling', True)
        self._precision = params.pop('precision', 1000)

        super(WeightedQLearning, self).__init__(self.Q, policy, gamma, params)

        self._n_updates = Table(shape)
        self._sigma = Table(shape, initial_value=1e10)
        self._Q = Table(shape)
        self._Q2 = Table(shape)
        self._weights_var = Table(shape)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]
        q_next = self._next_q(next_state) if not absorbing else 0.

        target = reward + self._gamma * q_next

        alpha = self.alpha(state, action)

        self.Q[state, action] = q_current + alpha * (target - q_current)

        self._n_updates[state, action] += 1

        self._Q[state, action] += (
            target - self._Q[state, action]) / self._n_updates[state, action]
        self._Q2[state, action] += (target ** 2. - self._Q2[
            state, action]) / self._n_updates[state, action]
        self._weights_var[state, action] = (
            1 - alpha) ** 2. * self._weights_var[state, action] + alpha ** 2.

        if self._n_updates[state, action] > 1:
            var = self._n_updates[state, action] * (
                self._Q2[state, action] - self._Q[state, action] ** 2.) / (
                self._n_updates[state, action] - 1.)
            var_estimator = var * self._weights_var[state, action]
            var_estimator = var_estimator if var_estimator >= 1e-10 else 1e-10
            self._sigma[state, action] = np.sqrt(var_estimator)

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
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'SpeedyQLearning'

        self.Q = Table(shape)
        self.old_q = deepcopy(self.Q)

        super(SpeedyQLearning, self).__init__(self.Q, policy, gamma, params)

    def _update(self, state, action, reward, next_state, absorbing):
        old_q = deepcopy(self.Q)

        max_q_cur = np.max(self.Q[next_state, :]) if not absorbing else 0.
        max_q_old = np.max(self.old_q[next_state, :]) if not absorbing else 0.

        target_cur = reward + self._gamma * max_q_cur
        target_old = reward + self._gamma * max_q_old

        alpha = self.alpha(state, action)
        q_cur = self.Q[state, action]
        self.Q[state, action] = q_cur + alpha * (target_old-q_cur) + (
            1. - alpha) * (target_cur - target_old)

        self.old_q = old_q


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'SARSA'

        self.Q = Table(shape)
        super(SARSA, self).__init__(self.Q, policy, gamma, params)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        self._next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self._next_action] if not absorbing else 0.

        self.Q[state, action] = q_current + self.alpha(
            state, action) * (reward + self._gamma * q_next - q_current)


class RLearning(TD):
    """
    R-Learning algorithm.
    "A Reinforcement Learning Method for Maximizing Undiscounted Rewards".
    Schwartz A.. 1993.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'RLearning'

        assert 'beta' in params['algorithm_params']

        self.Q = Table(shape)
        self._rho = 0.
        self.beta = params['algorithm_params']['beta']

        super(RLearning, self).__init__(self.Q, policy, gamma, params)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]
        q_next = np.max(self.Q[next_state, :]) if not absorbing else 0.
        delta = reward - self._rho + q_next - q_current
        q_new = q_current + self.alpha(state, action) * delta

        self.Q[state, action] = q_new

        q_max = np.max(self.Q[state, :])
        if q_new == q_max:
            delta = reward + q_next - q_max - self._rho
            self._rho += self.beta(state, action) * delta


class RQLearning(TD):
    """
    RQ-Learning algorithm.
    "Exploiting Structure and Uncertainty of Bellman Updates in Markov Decision
    Processes". Tateo D. et al.. 2017.

    """
    def __init__(self, shape, policy, gamma, params):
        self.__name__ = 'RQLearning'

        alg_params = params['algorithm_params']

        self.offpolicy = alg_params['offpolicy']
        if 'delta' in alg_params and 'beta' not in alg_params:
            self.delta = alg_params['delta']
            self.beta = None
        elif 'delta' not in alg_params and 'beta' in alg_params:
            self.delta = None
            self.beta = alg_params['beta']
        else:
            raise ValueError('delta or beta parameters needed.')

        self.Q = Table(shape)
        self.Q_tilde = Table(shape)
        self.R_tilde = Table(shape)
        super(RQLearning, self).__init__(self.Q, policy, gamma, params)

    def _update(self, state, action, reward, next_state, absorbing):
        alpha = self.alpha(state, action, target=reward)
        self.R_tilde[state, action] += alpha * (reward - self.R_tilde[
            state, action])

        if not absorbing:
            q_next = self._next_q(next_state)

            if self.delta is not None:
                beta = alpha * self.delta(state, action, target=q_next,
                                          factor=alpha)
            else:
                beta = self.beta(state, action, target=q_next)

            self.Q_tilde[state, action] += beta * (q_next - self.Q_tilde[
                state, action])

        q = self.R_tilde[state, action] + self.mdp_info['gamma'] * self.Q_tilde[
            state, action]
        self.Q[state, action] = q

    def _next_q(self, next_state):
        if self.offpolicy:
            return np.max(self.Q[next_state, :])
        else:
            self._next_action = self.draw_action(next_state)

            return self.Q[next_state, self._next_action]
