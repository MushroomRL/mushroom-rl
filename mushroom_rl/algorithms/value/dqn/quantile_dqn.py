from copy import deepcopy

import torch
import torch.nn.functional as F

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QuantileNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


class QuantileHuberLoss:
    """
    Quantile Huber loss for Quantile Regression DQN.

    """
    def __init__(self, n_quantiles):
        self._n_quantiles = n_quantiles
        self._tau_hat = self._build_tau_hat()

    def _build_tau_hat(self):
        tau = torch.arange(self._n_quantiles + 1, device=TorchUtils.get_device()) / self._n_quantiles
        return (tau[:-1] + tau[1:]) / 2

    def __call__(self, input, target, reduction='mean'):
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

    def __getstate__(self):
        return {'_n_quantiles': self._n_quantiles}

    def __setstate__(self, state):
        self._n_quantiles = state['_n_quantiles']
        self._tau_hat = self._build_tau_hat()


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

        self._n_quantiles = n_quantiles
        self._loss = QuantileHuberLoss(n_quantiles)

        params['approximator_params']['loss'] = self._loss

        self._add_save_attr(
            _n_quantiles='primitive',
            _loss='pickle'
        )

        super().__init__(mdp_info, policy, TorchApproximator, **params)

        self._fit_params['get_quantiles'] = True

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
