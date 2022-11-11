import numpy as np
from tqdm import trange

from mushroom_rl.algorithms.value.batch_td import BatchTD
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.parameters import to_parameter


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False):
        """
        Constructor.

        Args:
            n_iterations ([int, Parameter]): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = to_parameter(n_iterations)
        self._quiet = quiet
        self._target = None

        self._add_save_attr(
            _n_iterations='mushroom',
            _quiet='primitive',
            _target='pickle'
        )

        super().__init__(mdp_info, policy, approximator, approximator_params, fit_params)

    def fit(self, dataset, **info):
        state, action, reward, next_state, absorbing, _ = parse_dataset(dataset)
        for _ in trange(self._n_iterations(), dynamic_ncols=True, disable=self._quiet, leave=False):
            if self._target is None:
                self._target = reward
            else:
                q = self.approximator.predict(next_state)
                if np.any(absorbing):
                    q *= 1 - absorbing.reshape(-1, 1)

                max_q = np.max(q, axis=1)
                self._target = reward + self.mdp_info.gamma * max_q

            self.approximator.fit(state, action, self._target, **self._fit_params)
