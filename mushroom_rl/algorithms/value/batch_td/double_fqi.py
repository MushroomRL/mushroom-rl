import numpy as np
from tqdm import trange

from mushroom_rl.algorithms.value.batch_td.fqi import FQI


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning Problems"
    D'Eramo C. et al. 2017.

    """
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False, history_length=1):
        approximator_params['n_models'] = 2

        super().__init__(mdp_info, policy, approximator, n_iterations,
                         approximator_params, fit_params, quiet, history_length)

    def fit(self, dataset):
        self._history_manager.update_preprocessors(dataset)
        parsed = self._history_manager.parse_history(dataset)[:5]

        half = len(dataset) // 2
        state, action, reward, next_state, absorbing = [[x[i * half:(i + 1) * half] for i in range(2)] for x in parsed]

        for _ in trange(self._n_iterations(), dynamic_ncols=True, disable=self._quiet, leave=False):
            if self._target is None:
                self._target = list(reward)
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

            if self._logger:
                self._logger.advance_step()
