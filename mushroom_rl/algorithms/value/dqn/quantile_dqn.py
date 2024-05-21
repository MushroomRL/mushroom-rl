from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import NumpyTorchApproximator


def quantile_huber_loss(input, target):
    tau = QuantileDQN.tau_hat.repeat(input.shape[0], 1)

    target = target.t().unsqueeze(-1).repeat(1, 1, tau.shape[-1])
    input = input.repeat(tau.shape[-1], 1, 1)

    indicator = (((target - input) < 0.).type(torch.float))
    huber_loss = F.smooth_l1_loss(input, target, reduction='none')

    loss = torch.abs(tau - indicator) * huber_loss

    return loss.mean()


class QuantileNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_quantiles,
                 n_features, **kwargs):
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)
        self._n_quantiles = n_quantiles

        self._quant = nn.ModuleList(
            [nn.Linear(n_features, n_quantiles) for _ in range(self._n_output)])

        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._quant[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_quantiles=False):
        features = self._phi(state)

        a_quant = [self._quant[i](features) for i in range(self._n_output)]
        a_quant = torch.stack(a_quant, dim=1)

        if not get_quantiles:
            quant = a_quant.mean(-1)

            if action is not None:
                return torch.squeeze(quant.gather(1, action))
            else:
                return quant
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_quantiles)

                return torch.squeeze(a_quant.gather(1, action))
            else:
                return a_quant


class QuantileDQN(AbstractDQN):
    """
    Quantile Regression DQN algorithm.
    "Distributional Reinforcement Learning with Quantile Regression".
    Dabney W. et al.. 2018.

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
        params['approximator_params']['loss'] = quantile_huber_loss

        self._n_quantiles = n_quantiles

        tau = torch.arange(n_quantiles + 1) / n_quantiles
        QuantileDQN.tau_hat = torch.Tensor([(tau[i-1] + tau[i]) / 2 for i in range(1, len(tau))])

        self._add_save_attr(
            _n_quantiles='primitive'
        )

        super().__init__(mdp_info, policy, NumpyTorchApproximator, **params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self.target_approximator.predict(next_state, **self._predict_params)
            a_max = np.argmax(q_next, 1)
            quant_next = self.target_approximator.predict(next_state, a_max,
                                                          get_quantiles=True, **self._predict_params)
            quant_next *= (1 - absorbing).reshape(-1, 1)
            quant = reward.reshape(-1, 1) + self.mdp_info.gamma * quant_next

            self.approximator.fit(state, action, quant, get_quantiles=True, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
