from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.approximators.parametric.networks import QuantileNetwork


def quantile_huber_loss(input, target):
    tau = QuantileDQN.tau_hat.repeat(input.shape[0], 1)

    target = target.t().unsqueeze(-1).repeat(1, 1, tau.shape[-1])
    input = input.repeat(tau.shape[-1], 1, 1)

    indicator = (((target - input) < 0.).type(torch.float))
    huber_loss = F.smooth_l1_loss(input, target, reduction='none')

    loss = torch.abs(tau - indicator) * huber_loss

    return loss.mean()


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
            state, action, reward, next_state, absorbing, *_ =\
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
