import numpy as np
from tqdm import trange

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.features import get_action_features
from mushroom.utils.dataset import parse_dataset


class BatchTD(Agent):
    """
    Abstract class to implement a generic Batch TD algorithm.

    """
    def __init__(self, approximator, policy, mdp_info, params, features=None):
        """
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.

        """
        self._n_iterations = params['algorithm_params']['n_iterations']
        self._quiet = params['algorithm_params'].get('quiet', False)

        self.approximator = Regressor(approximator,
                                      **params['approximator_params'])
        policy.set_q(self.approximator)

        super(BatchTD, self).__init__(policy, mdp_info, params, features)


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, params):
        super(FQI, self).__init__(approximator, policy, mdp_info, params)

        self._target = None

        # "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.
        self._boosted = params['algorithm_params'].get('boosted', False)
        if self._boosted:
            self._prediction = 0.
            self._next_q = 0.
            self._idx = 0

    def fit(self, dataset, target=None):
        """
        Fit loop.

        Args:
            dataset (list): the dataset;
            target (np.ndarray, None): initial target of FQI.

        Returns:
            Last target computed.

        """
        if self._boosted:
            if self._target is None:
                self._prediction = 0.
                self._next_q = 0.
                self._idx = 0
            fit = self._fit_boosted
        else:
            fit = self._fit

        for _ in trange(self._n_iterations, dynamic_ncols=True,
                        disable=self._quiet, leave=False):
            fit(dataset)

    def _fit(self, x):
        """
        Single fit iteration.

        Args:
            x (list): the dataset.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if self._target is None:
            self._target = reward
        else:
            q = self.approximator.predict(next_state)
            if np.any(absorbing):
                q *= 1 - absorbing.reshape(-1, 1)

            max_q = np.max(q, axis=1)
            self._target = reward + self.mdp_info.gamma * max_q

        self.approximator.fit(state, action, self._target,
                              **self.params['fit_params'])

    def _fit_boosted(self, x):
        """
        Single fit iteration for boosted FQI.

        Args:
            x (list): the dataset.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if self._target is None:
            self._target = reward
        else:
            self._next_q += self.approximator.predict(next_state,
                                                      idx=self._idx - 1)
            if np.any(absorbing):
                self._next_q *= 1 - absorbing.reshape(-1, 1)

            max_q = np.max(self._next_q, axis=1)
            self._target = reward + self.mdp_info.gamma * max_q

        self._target -= self._prediction
        self._prediction += self._target

        self.approximator.fit(state, action, self._target, idx=self._idx,
                              **self.params['fit_params'])

        self._idx += 1


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et al.. 2017.

    """
    def __init__(self, approximator, policy, mdp_info, params):
        params['approximator_params']['n_models'] = 2
        super(DoubleFQI, self).__init__(approximator, policy, mdp_info, params)

    def _fit(self, x):
        state = list()
        action = list()
        reward = list()
        next_state = list()
        absorbing = list()

        half = len(x) / 2
        for i in xrange(2):
            s, a, r, ss, ab, _ = parse_dataset(x[i * half:(i + 1) * half])
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(ss)
            absorbing.append(ab)

        if self._target is None:
            self._target = reward
        else:
            for i in xrange(2):
                q_i = self.approximator.predict(next_state[i], idx=i)
                if np.any(absorbing[i]):
                    q_i *= 1 - absorbing[i].reshape(-1, 1)

                amax_q = np.expand_dims(np.argmax(q_i, axis=1), axis=1)
                max_q = self.approximator.predict(next_state[i], amax_q,
                                                  idx=1 - i)
                self._target[i] = reward[i] + self.mdp_info.gamma * max_q

        for i in xrange(2):
            self.approximator.fit(state[i], action[i], self._target[i], idx=i,
                                  **self.params['fit_params'])


class WeightedFQI(FQI):
    """
    Weighted Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et al.. 2017.

    """
    def _fit(self, x):
        pass


class LSPI(BatchTD):
    """
    Least-Squares Policy Iteration algorithm.
    "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

    """
    def __init__(self, policy, mdp_info, params, features):
        k = features.size * mdp_info.action_space.n
        self._A = np.zeros((k, k))
        self._b = np.zeros((k, 1))

        super(LSPI, self).__init__(LinearApproximator, policy, mdp_info, params,
                                   features)

    def fit(self, dataset):
        phi_state, action, reward, phi_next_state, absorbing, _ = parse_dataset(
            dataset, self.phi)
        phi_state_action = get_action_features(phi_state, action,
                                               self.mdp_info.action_space.n)
        q = self.approximator.predict(phi_next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        next_action = np.argmax(q, axis=1).reshape(-1, 1)
        phi_next_state_next_action = get_action_features(
            phi_next_state,
            next_action,
            self.mdp_info.action_space.n
        )

        tmp = phi_state_action - self.mdp_info.gamma *\
            phi_next_state_next_action
        self._A += phi_state_action.T.dot(tmp)
        self._b += (phi_state_action.T.dot(reward)).reshape(-1, 1)

        if np.linalg.matrix_rank(self._A) == self._A.shape[1]:
            w = np.linalg.solve(self._A, self._b)
        else:
            w = np.linalg.pinv(self._A).dot(self._b)
        self.approximator.set_weights(w)
