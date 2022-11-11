import numpy as np
from tqdm import trange

from mushroom_rl.utils.dataset import parse_dataset

from .fqi import FQI


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

    def fit(self, dataset, **info):
        for _ in trange(self._n_iterations(), dynamic_ncols=True, disable=self._quiet, leave=False):
            state = list()
            action = list()
            reward = list()
            next_state = list()
            absorbing = list()

            half = len(dataset) // 2
            for i in range(2):
                s, a, r, ss, ab, _ = parse_dataset(dataset[i * half:(i + 1) * half])
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
