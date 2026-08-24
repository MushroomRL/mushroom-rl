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

    def __call__(self, input, target):
        tau = self._tau_hat.repeat(input.shape[0], 1)

        target = target.t().unsqueeze(-1).repeat(1, 1, tau.shape[-1])
        input = input.repeat(tau.shape[-1], 1, 1)

        indicator = (((target - input) < 0.).type(torch.float))
        huber_loss = F.smooth_l1_loss(input, target, reduction='none')

        loss = torch.abs(tau - indicator) * huber_loss

        return loss.sum(-1).mean()

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

        params['approximator_params']['loss'] = QuantileHuberLoss(n_quantiles)

        self._add_save_attr(
            _n_quantiles='primitive'
        )

        super().__init__(mdp_info, policy, TorchApproximator, **params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_ =\
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = torch.clip(reward, -1, 1)

            with torch.no_grad():
                q_next = self.target_approximator.predict(next_state, **self._predict_params)
                a_max = torch.argmax(q_next, 1).unsqueeze(1)
                quant_next = self.target_approximator.predict(next_state, a_max, get_quantiles=True,
                                                              **self._predict_params)
                quant_next *= (~absorbing).unsqueeze(1)
                quant = reward.unsqueeze(1) + self.mdp_info.gamma * quant_next

            self.approximator.fit(state, action, quant, get_quantiles=True, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

            if self._logger:
                self._logger.advance_step()
