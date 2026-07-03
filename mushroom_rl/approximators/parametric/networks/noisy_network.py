import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mushroom_rl.utils.torch_utils import TorchUtils


class NoisyNetwork(nn.Module):
    """
    Network for Noisy DQN, outputting the Q-values through a noisy linear layer whose learnable noise provides
    state-dependent exploration.

    """
    class NoisyLinear(nn.Module):
        """
        Factorized noisy linear layer, adding learnable Gaussian noise to the weights and biases as described
        in "Noisy Networks for Exploration" by Fortunato et al.

        """
        __constants__ = ['in_features', 'out_features']

        def __init__(self, in_features, out_features, sigma_coeff=.5, bias=True):
            """
            Constructor.

            Args:
                in_features (int): size of each input sample;
                out_features (int): size of each output sample;
                sigma_coeff (float, .5): scaling coefficient for the initial noise standard deviation;
                bias (bool, True): whether to add a learnable (noisy) bias term.

            """
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
            eps_output = torch.rand(self.mu_weight.shape[0], 1, device=TorchUtils.get_device())
            eps_input = torch.rand(1, self.mu_weight.shape[1], device=TorchUtils.get_device())

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

    def __init__(self, input_shape, output_shape, features_network, n_features, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (the number of actions);
            features_network (nn.Module): the network used to compute the features;
            n_features (int): number of features extracted by the features network;
            **kwargs: parameters forwarded to the features network.

        """
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)

        self._Q = self.NoisyLinear(n_features, self._n_output)

    def forward(self, state, action=None):
        features = self._phi(state)
        q = self._Q(features)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted