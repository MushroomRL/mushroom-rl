import numpy as np

import torch
import torch.nn as nn

from mushroom_rl.policy import Policy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.torch_distributions import CategoricalWrapper, SquashedGaussian
from mushroom_rl.rl_utils.parameters import to_parameter

from itertools import chain


class TorchPolicy(Policy):
    """
    Interface for a generic PyTorch policy.
    A PyTorch policy is a policy implemented as a neural network using PyTorch.
    Its methods operate directly on torch tensors.

    """

    def __call__(self, state, action):
        return torch.exp(self.log_prob(state, action))

    def draw_with_log_prob(self, state):
        """
        Sample an action in ``state`` using the reparametrization trick and compute its log probability. Since the
        action is sampled through the reparametrization trick, gradients can flow through both the action and its
        log probability.

        Args:
            state (torch.Tensor): the set of states where the action is sampled.

        Returns:
            The sampled action and its log probability.

        """
        raise NotImplementedError

    def log_prob(self, state, action):
        """
        Compute the logarithm of the probability of taking ``action`` in ``state``.

        Args:
            state (torch.Tensor): set of states;
            action (torch.Tensor): set of actions.

        Returns:
            The tensor of log-probability.

        """
        raise NotImplementedError

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (torch.Tensor, None): the set of states to consider. If the entropy of the policy can be computed in
                closed form, then ``state`` can be None.

        Returns:
            The value of the entropy of the policy.

        """
        raise NotImplementedError

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (torch.Tensor): the set of states where the distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        raise NotImplementedError

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        raise NotImplementedError


class GaussianTorchPolicy(TorchPolicy):
    """
    Torch policy implementing a Gaussian policy with trainable standard
    deviation. The standard deviation is not state-dependent.

    """
    def __init__(self, network, input_shape, output_shape, std_0=1., **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            std_0 (float, 1.): initial standard deviation;
            **params: parameters used by the network constructor.

        """
        self._action_dim = output_shape[0]

        self._mu = TorchApproximator(input_shape=input_shape, output_shape=output_shape, network=network, **params)
        self._predict_params = dict()

        log_sigma_init = (torch.ones(self._action_dim, device=TorchUtils.get_device())
                          * torch.log(TorchUtils.to_float_tensor(std_0)))

        self._log_sigma = nn.Parameter(log_sigma_init)

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _log_sigma='torch'
        )

    def draw_action(self, state):
        with torch.no_grad():
            return self.distribution(state).sample()

    def draw_with_log_prob(self, state):
        dist = self.distribution(state)
        a = dist.rsample()

        return a, dist.log_prob(a).unsqueeze(-1)

    def log_prob(self, state, action):
        return self.distribution(state).log_prob(action).unsqueeze(-1)

    def entropy(self, state=None):
        return self._action_dim / 2 * torch.log(TorchUtils.to_float_tensor(2 * np.pi * np.e))\
               + torch.sum(self._log_sigma)

    def distribution(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def get_mean_and_chol(self, state):
        assert torch.all(torch.exp(self._log_sigma) > 0)
        return self._mu(state, **self._predict_params), torch.diag(torch.exp(self._log_sigma))

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


class BoltzmannTorchPolicy(TorchPolicy):
    """
    Torch policy implementing a Boltzmann policy.

    """
    def __init__(self, network, input_shape, output_shape, beta, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            beta ([float, Parameter]): the inverse of the temperature distribution. As
                the temperature approaches infinity, the policy becomes more and
                more random. As the temperature approaches 0.0, the policy becomes
                more and more greedy.
            **params: parameters used by the network constructor.

        """
        self._action_dim = output_shape[0]
        self._predict_params = dict()

        self._logits = TorchApproximator(input_shape=input_shape, output_shape=output_shape, network=network, **params)
        self._beta = to_parameter(beta)

        self._add_save_attr(
            _action_dim='primitive',
            _predict_params='pickle',
            _beta='mushroom',
            _logits='mushroom'
        )

    def draw_action(self, state):
        with torch.no_grad():
            return self.distribution(state).sample().unsqueeze(-1)

    def draw_with_log_prob(self, state):
        raise NotImplementedError("The Boltzmann policy cannot be sampled with the reparametrization trick.")

    def log_prob(self, state, action):
        return self.distribution(state).log_prob(action).unsqueeze(-1)

    def entropy(self, state=None):
        return torch.mean(self.distribution(state).entropy())

    def distribution(self, state):
        logits = self._logits(state, **self._predict_params) * self._beta(state.numpy())
        return CategoricalWrapper(logits)

    def set_weights(self, weights):
        self._logits.set_weights(weights)

    def get_weights(self):
        return self._logits.get_weights()

    def parameters(self):
        return self._logits.parameters()

    def set_beta(self, beta):
        self._beta = to_parameter(beta)


class SquashedGaussianTorchPolicy(TorchPolicy):
    """
    Torch policy implementing a Gaussian policy squashed by a tanh and remapped to a bounded action range, as used
    by the Soft Actor-Critic algorithm. The squashing and the corresponding change-of-variables are handled by the
    :class:`~mushroom_rl.utils.torch_distributions.SquashedGaussian` distribution.

    """
    def __init__(self, mu_approximator, sigma_approximator, min_a, max_a, log_std_min, log_std_max):
        """
        Constructor.

        Args:
            mu_approximator (Approximator): a regressor computing the mean given a state;
            sigma_approximator (Approximator): a regressor computing the log standard deviation given a state;
            min_a (np.ndarray): a vector specifying the minimum action value for each component;
            max_a (np.ndarray): a vector specifying the maximum action value for each component;
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

        self._min_a = TorchUtils.to_float_tensor(min_a)
        self._max_a = TorchUtils.to_float_tensor(max_a)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps = 1e-6

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _min_a='torch',
            _max_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps='primitive'
        )

    def draw_action(self, state):
        with torch.no_grad():
            return self.distribution(state).sample()

    def draw_with_log_prob(self, state):
        return self.distribution(state).rsample_and_log_prob()

    def log_prob(self, state, action):
        return self.distribution(state).log_prob(action).unsqueeze(-1)

    def entropy(self, state=None):
        _, log_prob = self.draw_with_log_prob(state)
        return -log_prob.mean()

    def distribution(self, state):
        mu = self._mu_approximator.predict(state)
        log_sigma = self._sigma_approximator.predict(state)
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return SquashedGaussian(mu, log_sigma.exp(), self._min_a, self._max_a, eps=self._eps)

    def set_weights(self, weights):
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return torch.concatenate([mu_weights, sigma_weights])

    def parameters(self):
        return chain(self._mu_approximator.parameters(), self._sigma_approximator.parameters())
