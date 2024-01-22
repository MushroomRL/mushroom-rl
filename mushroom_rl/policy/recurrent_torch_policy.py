import torch
import numpy as np

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.rl_utils.parameters import to_parameter


class RecurrentGaussianTorchPolicy(GaussianTorchPolicy):
    def __init__(self,  policy_state_shape, log_std_min=-20, log_std_max=2, **kwargs):

        super().__init__(policy_state_shape=policy_state_shape, **kwargs)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

    def reset(self):
        return torch.zeros(self.policy_state_shape)

    def draw_action(self, state, policy_state):
        with torch.no_grad():
            state = TorchUtils.to_float_tensor(state)
            policy_state = torch.as_tensor(policy_state)
            a, policy_state = self.draw_action_t(state, policy_state)
        return torch.squeeze(a, dim=0), policy_state

    def draw_action_t(self, state, policy_state):
        lengths = torch.tensor([1])
        state = torch.atleast_2d(state).view(1, 1, -1)
        policy_state = torch.atleast_2d(policy_state)

        dist, policy_state = self.distribution_and_policy_state_t(state, policy_state, lengths)
        action = dist.sample().detach()

        return action, policy_state

    def log_prob_t(self, state, action, policy_state, lengths):
        return self.distribution_t(state, policy_state, lengths).log_prob(action.squeeze())[:, None]

    def entropy_t(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def distribution(self, state, policy_state, lengths):
        s = TorchUtils.to_float_tensor(state)

        return self.distribution_t(s, policy_state, lengths)

    def distribution_t(self, state, policy_state, lengths):
        mu, sigma, _ = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def distribution_and_policy_state_t(self, state, policy_state, lengths):
        mu, sigma, policy_state = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma), policy_state

    def get_mean_and_covariance_and_policy_state(self, state, policy_state, lengths):
        mu, next_hidden_state = self._mu(state, policy_state, lengths, **self._predict_params)

        # Bound the log_std
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        covariance = torch.diag(torch.exp(2 * log_sigma))
        return mu, covariance, next_hidden_state
