import numpy as np
from tqdm import trange

from .fqi import FQI


class BoostedFQI(FQI):
    """
    Boosted Fitted Q-Iteration algorithm.
    "Boosted Fitted Q-Iteration". Tosatto S. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False):
        self._prediction = 0.
        self._next_q = 0.
        self._idx = 0

        assert approximator_params['n_models'] == n_iterations

        self._add_save_attr(
            _n_iterations='primitive',
            _quiet='primitive',
            _prediction='primitive',
            _next_q='numpy',
            _idx='primitive',
            _target='pickle'
        )

        super().__init__(mdp_info, policy, approximator, n_iterations, approximator_params, fit_params, quiet)

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, _ = dataset.parse(to='numpy')
        for _ in trange(self._n_iterations(), dynamic_ncols=True, disable=self._quiet, leave=False):
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
