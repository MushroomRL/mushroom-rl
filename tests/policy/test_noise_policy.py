import torch
import numpy as np

import pytest

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy, ClippedGaussianPolicy
from mushroom_rl.rl_utils.parameters import LinearParameter


def test_ornstein_uhlenbeck_policy():
    torch.manual_seed(42)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)
    assert torch.equal(pi.get_weights(), w)

    state = torch.randn(5)

    pi.reset()

    action = pi.draw_action(state)
    action_test = torch.tensor([-0.7055691481,  1.1255935431])
    assert torch.allclose(action, action_test)

    pi.reset()
    action = pi.draw_action(state)
    action_test = torch.tensor([-0.7114595175,  1.1141412258])
    assert torch.allclose(action, action_test)

    with pytest.raises(NotImplementedError):
        pi(state, action)


def test_ornstein_uhlenbeck_greedy():
    torch.manual_seed(42)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)

    state = torch.randn(5)

    pi.reset()

    action = pi.draw_action_greedy(state)
    action_test = torch.tensor([-0.7066923380, 1.1151391268])
    assert torch.allclose(action, action_test)

    assert torch.allclose(pi.draw_action_greedy(state), action_test)


def test_ornstein_uhlenbeck_vectorized():
    torch.manual_seed(42)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    ps = pi.reset_vectorized(torch.tensor([True, True, True]))
    assert ps.shape == (3, 2)
    assert torch.all(ps == 0.0)

    action = pi.draw_action(torch.randn(3, 5))
    assert action.shape == (3, 2)
    assert pi.policy_state.shape == (3, 2)

    before = pi.policy_state.clone()
    pi.reset_vectorized(torch.tensor([True, False, True]))
    assert torch.all(pi.policy_state[0] == 0.0)
    assert torch.all(pi.policy_state[2] == 0.0)
    assert torch.equal(pi.policy_state[1], before[1])


def test_clipped_gaussian_policy():
    torch.manual_seed(1)

    low = -torch.ones(2)
    high = torch.ones(2)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = ClippedGaussianPolicy(mu, torch.eye(2), low, high)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)
    assert torch.equal(pi.get_weights(), w)

    state = torch.randn(5)

    action = pi.draw_action(state)
    action_test = torch.tensor([-1.0, 1.0])
    assert torch.allclose(action, action_test)

    action = pi.draw_action(state)
    action_test = torch.tensor([0.4926533699, 1.0])
    assert torch.allclose(action, action_test)

    with pytest.raises(NotImplementedError):
        pi(state, action)


def test_clipped_gaussian_greedy():
    torch.manual_seed(1)

    low = -torch.ones(2)
    high = torch.ones(2)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = ClippedGaussianPolicy(mu, torch.eye(2), low, high)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)

    state = torch.randn(5)

    action = pi.draw_action_greedy(state)
    action_test = torch.tensor([-1.0, 1.0])
    assert torch.allclose(action, action_test)

    assert torch.allclose(pi.draw_action_greedy(state), action_test)


def test_clipped_gaussian_std_schedule(tmpdir):
    torch.manual_seed(1)

    low = -10 * torch.ones(2)
    high = 10 * torch.ones(2)
    std = LinearParameter(np.array([1., .5]), np.array([.1, .1]), n=2)

    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = ClippedGaussianPolicy(mu, std, low, high)
    pi.set_weights(torch.randn(pi.weights_size))

    state = torch.randn(5)

    action = pi.draw_action_greedy(state)
    action_test = torch.tensor([-2.0359039307, 2.8819246292])
    assert torch.allclose(action, action_test)

    actions = torch.stack([pi.draw_action(state) for _ in range(3)])
    actions_test = torch.tensor([[-2.2994337082, 3.2956271172],
                                 [-1.7830481529, 2.9229989052],
                                 [-2.1347045898, 2.7911171913]])
    assert torch.allclose(actions, actions_test)

    path = tmpdir / 'policy.msh'
    pi.save(path)
    pi_load = ClippedGaussianPolicy.load(path)

    torch.manual_seed(2)
    action = pi.draw_action(state)
    torch.manual_seed(2)
    action_load = pi_load.draw_action(state)
    assert torch.equal(action, action_load)


def test_clipped_gaussian_logs_sigma():
    class FakeLogger:
        def __init__(self):
            self.calls = list()

        def log_training(self, prefix=None, **kwargs):
            self.calls.append({(prefix + '/' + name if prefix else name): value for name, value in kwargs.items()})

    torch.manual_seed(1)
    mu = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    std = LinearParameter(np.array([1., .5]), np.array([.1, .1]), n=2)
    pi = ClippedGaussianPolicy(mu, std, -torch.ones(2), torch.ones(2))

    logger = FakeLogger()
    pi.set_logger(logger)
    pi.draw_action(torch.randn(5))
    pi.draw_action(torch.randn(5))

    assert len(logger.calls) == 2
    assert isinstance(logger.calls[0]['policy/sigma'], np.ndarray)
    assert np.allclose(logger.calls[0]['policy/sigma'], np.array([.55, .3]))
    assert np.allclose(logger.calls[1]['policy/sigma'], np.array([.1, .1]))
