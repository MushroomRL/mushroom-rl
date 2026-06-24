import torch
import numpy as np

from mushroom_rl.policy import GaussianTorchPolicy

from mushroom_rl.rl_utils.parameters import to_parameter


class RecurrentGaussianTorchPolicy(GaussianTorchPolicy):
    def __init__(self,  policy_state_shape, log_std_min=-20, log_std_max=2, **kwargs):

        super().__init__(policy_state_shape=policy_state_shape, n_outputs=2, **kwargs)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

    def reset(self):
        return torch.zeros(self.policy_state_shape)

    def draw_action(self, state, policy_state):
        with torch.no_grad():
            lengths = torch.tensor([1])
            state, policy_state = self._pad_state(state, policy_state)

            dist, policy_state = self.distribution_and_policy_state(state, policy_state, lengths)
            action = dist.sample()

            return action, policy_state

    def draw_with_log_prob(self, state, policy_state, lengths):
        dist, next_policy_state = self.distribution_and_policy_state(state, policy_state, lengths)
        action = dist.rsample()

        return action, dist.log_prob(action)[:, None], next_policy_state

    def log_prob(self, state, action, policy_state, lengths):
        return self.distribution(state, policy_state, lengths).log_prob(action)[:, None]

    def entropy(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def distribution(self, state, policy_state, lengths):
        mu, sigma, _ = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def distribution_and_policy_state(self, state, policy_state, lengths):
        mu, sigma, policy_state = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma), policy_state

    def get_mean_and_covariance_and_policy_state(self, state, policy_state, lengths):
        mu, next_hidden_state = self._mu(state, policy_state, lengths=lengths, **self._predict_params)

        # Bound the log_std
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        covariance = torch.diag(torch.exp(2 * log_sigma))
        return mu, covariance, next_hidden_state

    def _pad_state(self, state, policy_state):
        if state.ndim == len(self._mu.input_shape):
            state = state.unsqueeze(0)
            policy_state = policy_state.unsqueeze(0)
        state = state.unsqueeze(1)
        return state, policy_state
