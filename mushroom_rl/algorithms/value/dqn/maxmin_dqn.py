import numpy as np

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.regressor import Regressor


class MaxminDQN(AbstractDQN):
    """
    MaxminDQN algorithm.
    "Maxmin Q-learning: Controlling the Estimation Bias of Q-learning".
    Lan Q. et al.. 2020.

    """
    def __init__(self, mdp_info, policy, approximator, n_approximators,
                 **params):
        """
        Constructor.

        Args:
            n_approximators (int): the number of approximators in the ensemble.

        """
        assert n_approximators > 1

        self._n_approximators = n_approximators

        super().__init__(mdp_info, policy, approximator, **params)

    def _initialize_regressors(self, approximator, apprx_params_train,
                               apprx_params_target):
        self.approximator = Regressor(approximator,
                                      n_models=self._n_approximators,
                                      prediction='min',
                                      **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             n_models=self._n_approximators,
                                             prediction='min',
                                             **apprx_params_target)
        self._update_target()

    def fit(self, dataset):
        idx = np.random.randint(self._n_approximators)
        self._fit(dataset, approximator=self.approximator[idx])

        self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _update_target(self):
        for i in range(len(self.target_approximator)):
            self.target_approximator[i].set_weights(
                self.approximator[i].get_weights())
