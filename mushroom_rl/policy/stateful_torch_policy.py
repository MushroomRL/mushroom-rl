import torch
import torch.nn as nn

from itertools import chain

from mushroom_rl.policy.policy import StatefulPolicy
from mushroom_rl.policy.torch_policy import TorchPolicy
from mushroom_rl.approximators.parametric import RecurrentTorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.rl_utils.parameters import to_parameter


class StatefulTorchPolicy(StatefulPolicy, TorchPolicy):
    """
    Interface for a stateful PyTorch policy, i.e. a :class:`TorchPolicy` carrying a latent internal state (e.g. the
    hidden state of a recurrent network). ``draw_action`` relies on the stored internal state (see
    :class:`~mushroom_rl.policy.StatefulPolicy`), while the query methods take the policy state and the sequence
    ``lengths`` explicitly, so they never depend on the stored one.

    """
    def draw_with_log_prob(self, state, policy_state, lengths=None, **kwargs):
        """
        Sample an action using the reparametrization trick and compute its log probability.

        Args:
            state (torch.Tensor): the set of states where the action is sampled;
            policy_state (torch.Tensor): the policy internal states;
            lengths (torch.Tensor, None): the length of each input sequence, defaulted to ones when None;
            **kwargs: additional per-timestep conditioning inputs.

        Returns:
            The sampled action, its log probability and the next policy state.

        """
        raise NotImplementedError

    def log_prob(self, state, action, policy_state, lengths=None, **kwargs):
        """
        Compute the logarithm of the probability of taking ``action`` in ``state``.

        Args:
            state (torch.Tensor): set of states;
            action (torch.Tensor): set of actions;
            policy_state (torch.Tensor): the policy internal states;
            lengths (torch.Tensor, None): the length of each input sequence, defaulted to ones when None;
            **kwargs: additional per-timestep conditioning inputs.

        Returns:
            The tensor of log-probability.

        """
        raise NotImplementedError

    def distribution(self, state, policy_state, lengths=None, **kwargs):
        """
        Compute the policy distribution in the given states.

        Args:
            state (torch.Tensor): the set of states where the distribution is computed;
            policy_state (torch.Tensor): the policy internal states;
            lengths (torch.Tensor, None): the length of each input sequence, defaulted to ones when None;
            **kwargs: additional per-timestep conditioning inputs.

        Returns:
            The torch distribution for the provided states.

        """
        raise NotImplementedError


class RecurrentGaussianTorchPolicy(StatefulTorchPolicy):
    """
    Torch policy implementing a Gaussian policy whose mean is computed by a recurrent network. The hidden state of the
    network is the latent policy state, carried step-by-step at inference time and provided explicitly to the query
    methods together with the sequence ``lengths``.

    """
    def __init__(self, network, input_shape, output_shape, policy_state_shape, std_0=1.,
                 log_std_min=-20, log_std_max=2, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean regressor. Its
                ``forward`` must return ``(action_mean, next_policy_state)``;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space (the network internally also
                receives ``policy_state_shape`` as its second output shape);
            policy_state_shape (tuple): the shape of the hidden state of the recurrent network;
            std_0 (float, 1.): initial standard deviation;
            log_std_min ([float, Parameter], -20): min value for the policy log std;
            log_std_max ([float, Parameter], 2): max value for the policy log std;
            **params: parameters used by the network constructor.

        """
        super().__init__(policy_state_shape)

        self._action_dim = output_shape[0]

        self._mu = RecurrentTorchApproximator(input_shape=input_shape,
                                              output_shape=[output_shape, policy_state_shape],
                                              network=network, policy_state_shape=policy_state_shape, **params)
        self._predict_params = dict()

        log_sigma_init = torch.ones(self._action_dim, device=TorchUtils.get_device()) \
            * torch.log(TorchUtils.to_float_tensor(std_0))

        self._log_sigma = nn.Parameter(log_sigma_init)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _log_sigma='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom'
        )

    def draw_with_log_prob(self, state, policy_state, lengths=None, **kwargs):
        dist, next_policy_state = self.distribution_and_policy_state(state, policy_state, lengths, **kwargs)
        action = dist.rsample()

        return action, dist.log_prob(action)[:, None], next_policy_state

    def log_prob(self, state, action, policy_state, lengths=None, **kwargs):
        return self.distribution(state, policy_state, lengths, **kwargs).log_prob(action)[:, None]

    def entropy(self, state=None):
        return self._action_dim / 2 * torch.log(torch.tensor(2 * torch.pi * torch.e)) + torch.sum(self._log_sigma)

    def distribution(self, state, policy_state, lengths=None, **kwargs):
        mu, sigma, _ = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths, **kwargs)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def distribution_and_policy_state(self, state, policy_state, lengths=None, **kwargs):
        mu, sigma, policy_state = self.get_mean_and_covariance_and_policy_state(state, policy_state, lengths, **kwargs)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma), policy_state

    def get_mean_and_covariance_and_policy_state(self, state, policy_state, lengths=None, **kwargs):
        mu, next_hidden_state = self._mu(state, policy_state, lengths=lengths, **kwargs, **self._predict_params)

        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        covariance = torch.diag(torch.exp(2 * log_sigma))
        return mu, covariance, next_hidden_state

    def set_weights(self, weights):
        log_sigma_data = TorchUtils.to_float_tensor(weights[-self._action_dim:])
        self._log_sigma.data = log_sigma_data

        self._mu.set_weights(weights[:-self._action_dim])

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        sigma_weights = self._log_sigma.data.detach()

        return torch.concatenate([mu_weights, sigma_weights])

    def parameters(self):
        return chain(self._mu.parameters(), [self._log_sigma])

    def reset(self):
        self._policy_state = torch.zeros(self.policy_state_shape)

        return self._policy_state

    def reset_vectorized(self, start_mask):
        if self._policy_state is None:
            self._policy_state = torch.zeros((len(start_mask),) + self.policy_state_shape)
        self._policy_state[start_mask] = 0.

        return self._policy_state

    def _draw_action(self, state, policy_state, action_history=None):
        with torch.no_grad():
            dist, next_policy_state = self.distribution_and_policy_state(state, policy_state,
                                                                         action_history=action_history)
            action = dist.sample()

            return action, next_policy_state
