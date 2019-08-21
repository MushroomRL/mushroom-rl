import numpy as np

import torch
import torch.nn as nn

from mushroom.policy import Policy
from mushroom.utils.torch import get_weights, set_weights

from itertools import chain


class TorchPolicy(Policy):
    """
    Mushroom interface for a generic PyTorch policy.
    A PyTorch policy is a policy implemented as a neural network using PyTorch.
    """
    def __init__(self, use_cuda):
        """
        Constructor.

        Args:
            use_cuda (nn.Module): whether to use cuda or not.
        """
        self._use_cuda = use_cuda

    def __call__(self, state, action):
        s = self._to_tensor(state)
        a = self._to_tensor(action)

        return np.exp(self.log_prob_t(s, a).item())

    def draw_action(self, state):
        with torch.no_grad():
            s = torch.tensor(np.atleast_2d(state), dtype=torch.float)
            a = self.draw_action_t(s)

        return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def distribution(self, state):
        s = self._to_tensor(state)
        return self.distribution_t(s)

    def entropy(self, state=None):
        s = self._to_tensor(state) if state is not None else None
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
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def reset(self):
        pass

    def _to_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float)
        return x.cuda if self._use_cuda else x


class GaussianTorchPolicy(TorchPolicy):
    def __init__(self, network, input_shape, output_shape, std_0=1., use_cuda=False, **params):
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]

        self._mu = network(input_shape, output_shape, **params)
        self._log_sigma = nn.Parameter(torch.ones(self._action_dim) * np.log(std_0))

        if self._use_cuda:
            self._mu.cuda()
            self._log_sigma = self._log_sigma.cuda()

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
        return self._mu(state), torch.diag(torch.exp(2 * self._log_sigma))

    def set_weights(self, weights):
        self._log_sigma.data = torch.from_numpy(weights[-self._action_dim:])

        set_weights(self._mu.parameters(), weights[:-self._action_dim], self._use_cuda)

    def get_weights(self):
        mu_weights = get_weights(self._mu.parameters())
        sigma_weights = self._log_sigma.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, sigma_weights])

    def parameters(self):
        return chain(self._mu.parameters(), [self._log_sigma])
