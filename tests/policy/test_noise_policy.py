import torch

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy, ClippedGaussianPolicy


def test_ornstein_uhlenbeck_policy():
    torch.manual_seed(42)

    mu = Regressor(TorchApproximator, network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)
    assert torch.equal(pi.get_weights(), w)

    state = torch.randn(5)

    policy_state = pi.reset()

    action, policy_state = pi.draw_action(state, policy_state)
    action_test = torch.tensor([-0.7055691481,  1.1255935431])
    assert torch.allclose(action, action_test)

    policy_state = pi.reset()
    action, policy_state = pi.draw_action(state, policy_state)
    action_test = torch.tensor([-0.7114595175,  1.1141412258])
    assert torch.allclose(action, action_test)

    try:
        pi(state, action)
    except NotImplementedError:
        pass
    else:
        assert False


def test_clipped_gaussian_policy():
    torch.manual_seed(1)

    low = -torch.ones(2)
    high = torch.ones(2)

    mu = Regressor(TorchApproximator, network=LinearNetwork, input_shape=(5,), output_shape=(2,))
    pi = ClippedGaussianPolicy(mu, torch.eye(2), low, high)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)
    assert torch.equal(pi.get_weights(), w)

    state = torch.randn(5)

    action, _ = pi.draw_action(state)
    action_test = torch.tensor([-1.0, 1.0])
    assert torch.allclose(action, action_test)

    action, _ = pi.draw_action(state)
    action_test = torch.tensor([0.4926533699, 1.0])
    assert torch.allclose(action, action_test)

    try:
        pi(state, action)
    except NotImplementedError:
        pass
    else:
        assert False

# TODO Missing test for clipped gaussian!

