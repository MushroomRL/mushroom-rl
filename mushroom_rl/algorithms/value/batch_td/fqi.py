import numpy as np
from tqdm import trange

from mushroom_rl.algorithms.value.batch_td import BatchTD
from mushroom_rl.utils.dataset import parse_dataset


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False,
                 boosted=False):
        """
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not;
            boosted (bool, False): whether to use boosted FQI or not.

        """
        self._n_iterations = n_iterations
        self._quiet = quiet

        # "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.
        self._boosted = boosted
        if self._boosted:
            self._prediction = 0.
            self._next_q = 0.
            self._idx = 0
            approximator_params['n_models'] = n_iterations

        self._add_save_attr(
            _n_iterations='primitive',
            _quiet='primitive',
            _boosted='primitive',
            _prediction='primitive',
            _next_q='numpy',
            _idx='primitive',
            _target='pickle'
        )

        super().__init__(mdp_info, policy, approximator, approximator_params,
                         fit_params)

        self._target = None

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
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False):
        approximator_params['n_models'] = 2

        super().__init__(mdp_info, policy, approximator, n_iterations,
                         approximator_params, fit_params, quiet)

    def _fit(self, x):
        state = list()
        action = list()
        reward = list()
        next_state = list()
        absorbing = list()

        half = len(x) // 2
        for i in range(2):
            s, a, r, ss, ab, _ = parse_dataset(x[i * half:(i + 1) * half])
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(ss)
            absorbing.append(ab)

        if self._target is None:
            self._target = reward
        else:
            for i in range(2):
                q_i = self.approximator.predict(next_state[i], idx=i)

                amax_q = np.expand_dims(np.argmax(q_i, axis=1), axis=1)
                max_q = self.approximator.predict(next_state[i], amax_q,
                                                  idx=1 - i)
                if np.any(absorbing[i]):
                    max_q *= 1 - absorbing[i]
                self._target[i] = reward[i] + self.mdp_info.gamma * max_q

        for i in range(2):
            self.approximator.fit(state[i], action[i], self._target[i], idx=i,
                                  **self._fit_params)
