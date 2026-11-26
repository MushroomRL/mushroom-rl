import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
        self._sigma_coeff = sigma_coeff

        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
        fan_in **= .5
        bound_weight = 1 / fan_in
        bound_sigma = self._sigma_coeff / fan_in

        nn.init.uniform_(self.mu_weight, -bound_weight, bound_weight)
        nn.init.constant_(self.sigma_weight, bound_sigma)
        if self.mu_bias is not None:
            nn.init.uniform_(self.mu_bias, -bound_weight, bound_weight)
            nn.init.constant_(self.sigma_bias, bound_sigma)

    def forward(self, input):
        eps_output = torch.randn(self.mu_weight.shape[0], 1, device=self.mu_weight.device)
        eps_input = torch.randn(1, self.mu_weight.shape[1], device=self.mu_weight.device)

        eps_dot = torch.matmul(self._noise(eps_output), self._noise(eps_input))
        weight = self.mu_weight + self.sigma_weight * eps_dot

        if self.mu_bias is not None:
            bias = self.mu_bias + self.sigma_bias * self._noise(eps_output[:, 0])
        else:
            bias = None

        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.mu_bias is not None
        )

    @staticmethod
    def _noise(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))
