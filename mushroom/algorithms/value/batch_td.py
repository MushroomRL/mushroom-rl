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
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 fit_params=None, approximator_params=None, features=None,
                 quiet=False):
        """
        Constructor.

        Args:
            approximator (object): approximator used by the algorithm and the
                policy.
            n_iterations (int): number of iterations to perform for training;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            approximator_params (dict, None): parameters of the approximator to
                build;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._fit_params = dict() if fit_params is None else fit_params
        self._approximator_params = dict() if approximator_params is None else\
            approximator_params
        self._quiet = quiet

        self.approximator = Regressor(approximator,
                                      **self._approximator_params)
        policy.set_q(self.approximator)

        super(BatchTD, self).__init__(policy, mdp_info, features)


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 fit_params=None, approximator_params=None, features=None,
                 quiet=False, boosted=False):
        """
        Constructor.

        Args:
            boosted (bool, False): whether to use boosted FQI or not.

        """
        super(FQI, self).__init__(approximator, policy, mdp_info, n_iterations,
                                  fit_params, approximator_params, features,
                                  quiet)

        self._target = None

        # "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.
        self._boosted = boosted
        if self._boosted:
            self._prediction = 0.
            self._next_q = 0.
            self._idx = 0

    def fit(self, dataset):
        """
        Fit loop.

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

        self.approximator.fit(state, action, self._target, **self._fit_params)

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
                              **self._fit_params)

        self._idx += 1


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et al.. 2017.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 fit_params=None, approximator_params=None, features=None,
                 quiet=False):
        super(DoubleFQI, self).__init__(approximator, policy, mdp_info,
                                        n_iterations, fit_params,
                                        approximator_params, features, quiet)

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

                amax_q = np.expand_dims(np.argmax(q_i, axis=1), axis=1)
                max_q = self.approximator.predict(next_state[i], amax_q,
                                                  idx=1 - i)
                if np.any(absorbing[i]):
                    max_q *= 1 - absorbing[i]
                self._target[i] = reward[i] + self.mdp_info.gamma * max_q

        for i in xrange(2):
            self.approximator.fit(state[i], action[i], self._target[i], idx=i,
                                  **self._fit_params)


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
