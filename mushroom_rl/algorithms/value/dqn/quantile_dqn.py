from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QuantileNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


class QuantileHuberLoss(nn.Module):
    """
    Quantile Huber loss for Quantile Regression DQN.

    """
    def __init__(self, n_quantiles):
        """
        Constructor.

        Args:
            n_quantiles (int): number of quantiles.

        """
        super().__init__()

        tau = torch.arange(n_quantiles + 1, device=TorchUtils.get_device()) / n_quantiles
        self.register_buffer('_tau_hat', (tau[:-1] + tau[1:]) / 2)

    def forward(self, input, target, reduction='mean'):
        tau = self._tau_hat.repeat(input.shape[0], 1)

        target = target.t().unsqueeze(-1).repeat(1, 1, tau.shape[-1])
        input = input.repeat(tau.shape[-1], 1, 1)

        indicator = (((target - input) < 0.).type(torch.float))
        huber_loss = F.smooth_l1_loss(input, target, reduction='none')

        loss = torch.abs(tau - indicator) * huber_loss
        per_sample_loss = loss.sum(-1).mean(0)

        if reduction == 'mean':
            return per_sample_loss.mean()
        elif reduction == 'none':
            return per_sample_loss
        else:
            raise ValueError


class QuantileDQN(AbstractDQN):
    """
    Quantile Regression DQN algorithm.
    "Distributional Reinforcement Learning with Quantile Regression"
    Dabney W. et al. 2018.

    """
    def __init__(self, mdp_info, policy, approximator_params, n_quantiles, **params):
        """
        Constructor.

        Args:
            n_quantiles (int): number of quantiles.

        """
        features_network = approximator_params['network']
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = QuantileNetwork
        params['approximator_params']['features_network'] = features_network
        params['approximator_params']['n_quantiles'] = n_quantiles

        params['approximator_params']['loss'] = QuantileHuberLoss(n_quantiles)

        super().__init__(mdp_info, policy, TorchApproximator, **params)

        self._n_quantiles = n_quantiles
        self._loss = self.approximator.loss

        self._fit_params['get_quantiles'] = True

        self._add_save_attr(
            _n_quantiles='primitive',
            _loss='none'
        )

    def _post_load(self, device):
        super()._post_load(device)

        self._loss = self.approximator.loss

    def _compute_target(self, reward, next_state, absorbing):
        q_next = self.target_approximator.predict(next_state, **self._predict_params)
        a_max = torch.argmax(q_next, 1).unsqueeze(1)
        quant_next = self.target_approximator.predict(next_state, a_max, get_quantiles=True,
                                                      **self._predict_params)
        quant_next *= (~absorbing).unsqueeze(1)

        return reward.unsqueeze(1) + self.mdp_info.gamma * quant_next

    def _compute_priority(self, state, action, target):
        pred = self.approximator.predict(state, action, get_quantiles=True, **self._predict_params)

        return self._loss(pred, target, reduction='none')
