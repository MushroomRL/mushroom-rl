import torch

from mushroom_rl.algorithms.value.dqn import AbstractDQN


class AveragedDQN(AbstractDQN):
    """
    Averaged-DQN algorithm.
    "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning".
    Anschel O. et al. 2017.

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
        self.approximator = approximator(**apprx_params_train)
        self.target_approximator = approximator(n_models=self._n_approximators, prediction=None,
                                                **apprx_params_target)
        w = self.approximator.get_weights()
        self.target_approximator.set_weights(w.repeat(self._n_approximators, 1))

    def _update_target(self):
        idx = self._n_updates // self._target_update_frequency % self._n_approximators
        self.target_approximator[idx].set_weights(self.approximator.get_weights())

        if self._n_fitted_target_models < self._n_approximators:
            self._n_fitted_target_models += 1

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        q = q[:self._n_fitted_target_models].mean(0)
        if absorbing.any():
            q *= ~absorbing.unsqueeze(1)

        return q.max(1).values
