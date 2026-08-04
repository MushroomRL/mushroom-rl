import torch

from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric import TorchApproximator


class MaxminDQN(DQN):
    """
    MaxminDQN algorithm.
    "Maxmin Q-learning: Controlling the Estimation Bias of Q-learning"
    Lan Q. et al. 2020.

    """
    def __init__(self, mdp_info, policy, approximator=TorchApproximator, n_approximators=2, **params):
        """
        Constructor.

        Args:
            approximator (class, TorchApproximator): the approximator to use to fit the Q-function;
            n_approximators (int, 2): the number of approximators in the ensemble.

        """
        assert n_approximators > 1

        self._n_approximators = n_approximators

        super().__init__(mdp_info, policy, approximator, **params)

    def fit(self, dataset):
        self._fit_params['idx'] = torch.randint(self._n_approximators, (1,)).item()

        super().fit(dataset)

    def _initialize_regressors(self, approximator, apprx_params_train, apprx_params_target):
        self.approximator = approximator(n_models=self._n_approximators, prediction='min', **apprx_params_train)
        self.target_approximator = approximator(n_models=self._n_approximators, prediction='min',
                                                **apprx_params_target)
        self._update_target()

    def _update_target(self):
        self.target_approximator.set_weights(self.approximator.get_weights())
