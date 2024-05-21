import numpy as np

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.regressor import Regressor


class AveragedDQN(AbstractDQN):
    """
    Averaged-DQN algorithm.
    "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement
    Learning". Anschel O. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, approximator, n_approximators,
                 **params):
        """
        Constructor.

        Args:
            n_approximators (int): the number of target approximators to store.

        """
        assert n_approximators > 1

        self._n_approximators = n_approximators

        super().__init__(mdp_info, policy, approximator, **params)

        self._n_fitted_target_models = 1

        self._add_save_attr(_n_fitted_target_models='primitive')

    def _initialize_regressors(self, approximator, apprx_params_train,
                               apprx_params_target):
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             n_models=self._n_approximators,
                                             **apprx_params_target)
        for i in range(len(self.target_approximator)):
            self.target_approximator[i].set_weights(
                self.approximator.get_weights()
            )

    def _update_target(self):
        idx = self._n_updates // self._target_update_frequency\
              % self._n_approximators
        self.target_approximator[idx].set_weights(
            self.approximator.get_weights())

        if self._n_fitted_target_models < self._n_approximators:
            self._n_fitted_target_models += 1

    def _next_q(self, next_state, absorbing):
        q = list()
        for idx in range(self._n_fitted_target_models):
            q_target_idx = self.target_approximator.predict(next_state, idx=idx, **self._predict_params)
            q.append(q_target_idx)
        q = np.mean(q, axis=0)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)
