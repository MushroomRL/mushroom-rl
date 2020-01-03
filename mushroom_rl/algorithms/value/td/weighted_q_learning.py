import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table


class WeightedQLearning(TD):
    """
    Weighted Q-Learning algorithm.
    "Estimating the Maximum Expected Value through Gaussian Approximation".
    D'Eramo C. et. al.. 2016.

    """
    def __init__(self, mdp_info, policy, learning_rate, sampling=True,
                 precision=1000):
        """
        Constructor.

        Args:
            sampling (bool, True): use the approximated version to speed up
                the computation;
            precision (int, 1000): number of samples to use in the approximated
                version.

        """
        self.Q = Table(mdp_info.size)
        self._sampling = sampling
        self._precision = precision

        super().__init__(mdp_info, policy, self.Q, learning_rate)

        self._n_updates = Table(mdp_info.size)
        self._sigma = Table(mdp_info.size, initial_value=1e10)
        self._Q = Table(mdp_info.size)
        self._Q2 = Table(mdp_info.size)
        self._weights_var = Table(mdp_info.size)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]
        q_next = self._next_q(next_state) if not absorbing else 0.

        target = reward + self.mdp_info.gamma * q_next

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
            var_estimator = np.maximum(var_estimator, 1e-10)
            self._sigma[state, action] = np.sqrt(var_estimator)

    def _next_q(self, next_state):
        """
        Args:
            next_state (np.ndarray): the state where next action has to be
                evaluated.

        Returns:
            The weighted estimator value in ``next_state``.

        """
        means = self.Q[next_state, :]
        sigmas = np.zeros(self.Q.shape[-1])

        for a in range(sigmas.size):
            sigmas[a] = self._sigma[next_state, np.array([a])]

        if self._sampling:
            samples = np.random.normal(np.repeat([means], self._precision, 0),
                                       np.repeat([sigmas], self._precision, 0))
            max_idx = np.argmax(samples, axis=1)
            max_idx, max_count = np.unique(max_idx, return_counts=True)
            count = np.zeros(means.size)
            count[max_idx] = max_count

            self._w = count / self._precision
        else:
            raise NotImplementedError

        return np.dot(self._w, means)
