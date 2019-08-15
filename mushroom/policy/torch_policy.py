import numpy as np

import torch
import torch.nn as nn

from mushroom.policy import Policy
from mushroom.utils.torch import get_weights, set_weights


class TorchPolicy(Policy):
    """
    Mushroom interface for a generic PyTorch policy.
    A PyTorch policy is a policy implemented as a neural network using PyTorch.
    """
    def __init__(self, network, input_shape, output_shape, use_cuda, params):
        """
        Constructor.

        Args:
            network (nn.Module): the Torch policy. Must provide the forward interface
               to sample actions and the log_prob interface to compute the probability
               of a given state action pair.
        """
        self._network = network(input_shape, output_shape, use_cuda=use_cuda, **params)
        self._use_cuda = use_cuda

    def __call__(self, state, action):
        s = torch.tensor(state, dtype=torch.float)
        a = torch.tensor(action, dtype=torch.float)

        return np.exp(self.log_prob_t(s, a).item())

    def draw_action(self, state):
        with torch.no_grad():
            s = torch.tensor(np.atleast_2d(state), dtype=torch.float)
            a = self.draw_action_t(s)
        return torch.squeeze(a, dim=0).detach().numpy()

    def distribution(self, state):
        s = torch.tensor(state, dtype=torch.float)
        return self.distribution_t(s)

    def entropy(self, state=None):
        s = torch.tensor(state, dtype=torch.float) if state is not None else None
        return self.entropy_t(s)

    def draw_action_t(self, state):
        raise NotImplementedError

    def log_prob_t(self, state, action):
        raise NotImplementedError

    def entropy_t(self, state):
        raise NotImplementedError

    def distribution_t(self, state):
        raise NotImplementedError

    def set_weights(self, weights):
        set_weights(self._network.parameters(), weights, self._use_cuda)

    def get_weights(self):
        return get_weights(self._network.parameters())

    def parameters(self):
        return self._network.parameters()

    def reset(self):
        pass


class GaussianTorchPolicy(TorchPolicy):
    def __init__(self, network, input_shape, output_shape, std_0=1., use_cuda=False, **params):
        self._action_dim = output_shape[0]
        self._log_sigma = nn.Parameter(torch.ones(self._action_dim) * np.log(std_0))

        super().__init__(network, input_shape, output_shape, use_cuda, params)

    def draw_action_t(self, state):
        return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma).detach().numpy()

    def distribution_t(self, state):
        mu, sigma = self.get_mean_and_covariance(state)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def get_mean_and_covariance(self, state):
        return self._network(state), torch.diag(torch.exp(2 * self._log_sigma))
