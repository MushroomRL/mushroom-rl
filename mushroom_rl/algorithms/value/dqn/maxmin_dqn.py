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

        params['approximator_params'] = dict(params['approximator_params'])
        params['approximator_params']['n_models'] = n_approximators
        params['approximator_params']['prediction'] = 'min'

        super().__init__(mdp_info, policy, approximator, **params)

    def fit(self, dataset):
        self._fit_params['idx'] = torch.randint(len(self.approximator), (1,)).item()

        super().fit(dataset)
