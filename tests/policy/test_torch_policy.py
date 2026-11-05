import torch
import torch.nn as nn

import numpy as np

import pytest

from helper.cases import needs_cuda

from mushroom_rl.policy.torch_policy import TorchPolicy, GaussianTorchPolicy, BoltzmannTorchPolicy, \
    SquashedGaussianTorchPolicy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter


def abstract_method_tester(f, *args):
    with pytest.raises(NotImplementedError):
        f(*args)


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, -1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)

        return a


def test_interface():
    tmp = TorchPolicy()
    abstract_method_tester(tmp.draw_with_log_prob, None)
    abstract_method_tester(tmp.log_prob, None, None)
    abstract_method_tester(tmp.entropy, None)
    abstract_method_tester(tmp.distribution, None)
    abstract_method_tester(tmp.set_weights, None)
    abstract_method_tester(tmp.get_weights)
    abstract_method_tester(tmp.parameters)
    tmp.reset()


def test_gaussian():
    np.random.seed(42)
    torch.manual_seed(42)
    pi = GaussianTorchPolicy(Network, (3,), (2,), n_features=50)

    state = torch.as_tensor(np.random.rand(3))
    action = pi.draw_action(state)
    action_test = np.array([-2.242642,  1.444643])
    assert np.allclose(action.detach().cpu().numpy(), action_test)

    p_sa = pi(state, torch.as_tensor(action))
    p_sa_test = 0.04761083796620369
    assert np.allclose(p_sa.detach().cpu().numpy(), p_sa_test)

    entropy = pi.entropy()
    entropy_test = 2.837877035140991
    assert np.allclose(entropy.item(), entropy_test)

    np.random.seed(42)
    torch.manual_seed(42)
    pi = GaussianTorchPolicy(Network, (3,), (2,), n_features=50)

    state = torch.as_tensor(np.random.rand(4, 3))

    action = pi.draw_action_greedy(state)
    assert action.shape == (4, 2)
    assert not action.requires_grad
    assert torch.allclose(action, pi.distribution(state).mean)
    assert torch.allclose(action, pi.draw_action_greedy(state))


def test_boltzmann():
    np.random.seed(42)
    torch.manual_seed(42)
    beta = Parameter(1.0)
    pi = BoltzmannTorchPolicy(Network, (3,), (2,), beta, n_features=50)

    state = torch.as_tensor(np.random.rand(3, 3))
    action = pi.draw_action(state)
    action_test = np.array([[1], [0], [0]])
    assert np.allclose(action.detach().cpu().numpy(), action_test)

    p_sa = pi(state[0], action[0])
    p_sa_test = 0.8161248564720154
    assert np.allclose(p_sa.detach(), p_sa_test)

    states = torch.as_tensor(np.random.rand(1000, 3))
    entropy = pi.entropy(states)
    entropy_test = 0.5904456377029419
    assert np.allclose(entropy.item(), entropy_test)

    abstract_method_tester(pi.draw_with_log_prob, state)

    np.random.seed(42)
    torch.manual_seed(42)
    beta = Parameter(1.0)
    pi = BoltzmannTorchPolicy(Network, (3,), (2,), beta, n_features=50)

    state = torch.as_tensor(np.random.rand(3, 3))

    action = pi.draw_action_greedy(state)
    assert action.shape == (3, 1)
    assert torch.equal(action, pi.distribution(state).mode.unsqueeze(-1))
    assert torch.equal(action, pi.draw_action_greedy(state))

    np.random.seed(42)
    torch.manual_seed(42)
    beta = LinearParameter(value=5., threshold_value=1., n=100)
    pi = BoltzmannTorchPolicy(Network, (3,), (2,), beta, n_features=50)

    state = torch.as_tensor(np.random.rand(3, 3))

    for _ in range(50):
        pi.draw_action_greedy(state)
    pi.entropy(state)
    pi.log_prob(state, torch.tensor([[0], [1], [0]]))

    assert beta.get_value() == 5.

    for _ in range(10):
        pi.draw_action(state)

    assert np.allclose(beta.get_value(), 4.6)


def test_squashed_gaussian():
    np.random.seed(42)
    torch.manual_seed(42)

    mu_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    sigma_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    min_a = np.array([-2., -2.])
    max_a = np.array([2., 2.])
    pi = SquashedGaussianTorchPolicy(mu_approximator, sigma_approximator, min_a, max_a, -20, 2)

    state = torch.as_tensor(np.random.rand(5, 3))

    torch.manual_seed(42)
    action = pi.draw_action(state)
    action_test = np.load('tests/policy/data/test_squashed_gaussian_action.npy')
    assert action.shape == (5, 2)
    assert not action.requires_grad
    assert torch.all(action >= torch.as_tensor(min_a)) and torch.all(action <= torch.as_tensor(max_a))
    assert np.allclose(action.cpu().numpy(), action_test, atol=1e-5)

    torch.manual_seed(42)
    action_new, log_prob = pi.draw_with_log_prob(state)
    log_prob_test = np.load('tests/policy/data/test_squashed_gaussian_log_prob.npy')
    assert action_new.requires_grad and log_prob.requires_grad
    assert log_prob.shape == (5,)
    assert np.allclose(log_prob.detach().cpu().numpy(), log_prob_test, atol=1e-5)

    log_prob_ext = pi.log_prob(state, action_new.detach())
    assert log_prob_ext.shape == (5, 1)
    assert np.allclose(log_prob_ext.detach().cpu().numpy().squeeze(-1), log_prob_test, atol=1e-4)

    entropy = pi.entropy(state)
    entropy_test = 1.5094861
    assert entropy.requires_grad
    assert np.allclose(entropy.item(), entropy_test, atol=1e-5)

    assert pi.distribution(state).__class__.__name__ == 'SquashedGaussian'

    np.random.seed(42)
    torch.manual_seed(42)

    mu_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    sigma_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    min_a = np.array([-2., -2.])
    max_a = np.array([2., 2.])
    pi = SquashedGaussianTorchPolicy(mu_approximator, sigma_approximator, min_a, max_a, -20, 2)

    state = torch.as_tensor(np.random.rand(5, 3))

    action = pi.draw_action_greedy(state)
    assert action.shape == (5, 2)
    assert not action.requires_grad
    assert torch.all(action >= torch.as_tensor(min_a)) and torch.all(action <= torch.as_tensor(max_a))
    assert torch.allclose(action, pi.distribution(state).median)
    assert torch.allclose(action, pi.draw_action_greedy(state))


@needs_cuda()
def test_gaussian_to_device():
    pi = GaussianTorchPolicy(Network, (3,), (2,), n_features=50, device='cpu')
    parameters = list(pi.parameters())

    assert pi.to('cuda') is pi

    assert all(old is new for old, new in zip(parameters, pi.parameters()))
    assert isinstance(pi._log_sigma, torch.nn.Parameter)
    assert {p.device.type for p in pi.parameters()} == {'cuda'}
    assert pi.draw_action(torch.ones(3, device='cuda'))[0].device.type == 'cuda'


def test_weights():
    torch.manual_seed(1)
    state = torch.rand(4, 3)
    gaussian = GaussianTorchPolicy(Network, (3,), (2,), n_features=50)
    action, log_prob = gaussian.draw_with_log_prob(state)

    assert action.shape == (4, 2) and log_prob.shape == (4, 1)
    assert torch.allclose(log_prob, gaussian.log_prob(state, action.detach()))

    boltzmann = BoltzmannTorchPolicy(Network, (3,), (2,), Parameter(1.), n_features=50)
    weights = boltzmann.get_weights()
    boltzmann.set_weights(torch.zeros_like(weights))
    boltzmann.set_beta(2.)

    assert torch.equal(boltzmann.get_weights(), torch.zeros_like(weights))
    assert len(list(boltzmann.parameters())) == 6
    assert boltzmann._beta.get_value() == 2.

    mu_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    sigma_approximator = TorchApproximator(input_shape=(3,), output_shape=(2,), network=Network, n_features=50)
    squashed = SquashedGaussianTorchPolicy(mu_approximator, sigma_approximator, np.array([-2., -2.]),
                                           np.array([2., 2.]), -20, 2)
    weights = torch.arange(len(squashed.get_weights()), dtype=torch.float32)
    squashed.set_weights(weights)

    assert torch.equal(squashed.get_weights(), weights)
