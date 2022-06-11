from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric import TorchApproximator


class NoisyNetwork(nn.Module):
    class NoisyLinear(nn.Module):
        __constants__ = ['in_features', 'out_features']

        def __init__(self, in_features, out_features, use_cuda, sigma_coeff=.5, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
            self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.mu_bias = Parameter(torch.Tensor(out_features))
                self.sigma_bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self._use_cuda = use_cuda
            self._sigma_coeff = sigma_coeff

            self.reset_parameters()

        def reset_parameters(self):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            fan_in **= .5
            bound_weight = 1 / fan_in
            bound_sigma = self._sigma_coeff / fan_in

            nn.init.uniform_(self.mu_weight, -bound_weight, bound_weight)
            nn.init.constant_(self.sigma_weight, bound_sigma)
            if hasattr(self, 'mu_bias'):
                nn.init.uniform_(self.mu_bias, -bound_weight, bound_weight)
                nn.init.constant_(self.sigma_bias, bound_sigma)

        def forward(self, input):
            eps_output = torch.rand(self.mu_weight.shape[0], 1)
            eps_input = torch.rand(1, self.mu_weight.shape[1])
            if self._use_cuda:
                eps_output = eps_output.cuda()
                eps_input = eps_input.cuda()
            eps_dot = torch.matmul(self._noise(eps_output), self._noise(eps_input))
            weight = self.mu_weight + self.sigma_weight * eps_dot

            if hasattr(self, 'mu_bias'):
                self.bias = self.mu_bias + self.sigma_bias * self._noise(eps_output[:, 0])

            return F.linear(input, weight, self.bias)

        @staticmethod
        def _noise(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        def extra_repr(self):
            return 'in_features={}, out_features={}, mu_bias={}, sigma_bias={}'.format(
                self.in_features, self.out_features, self.mu_bias, self.sigma_bias is not None
            )

    def __init__(self, input_shape, output_shape, features_network, n_features, use_cuda, **kwargs):
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)

        self._Q = self.NoisyLinear(n_features, self._n_output, use_cuda)

    def forward(self, state, action=None):
        features = self._phi(state)
        q = self._Q(features)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class NoisyDQN(DQN):
    """
    Noisy DQN algorithm.
    "Noisy networks for exploration".
    Fortunato M. et al.. 2018.

    """
    def __init__(self, mdp_info, policy, approximator_params, **params):
        """
        Constructor.

        """
        features_network = approximator_params['network']
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = NoisyNetwork
        params['approximator_params']['features_network'] = features_network

        super().__init__(mdp_info, policy, TorchApproximator, **params)
