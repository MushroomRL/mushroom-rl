import torch

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy


def test_ornstein_uhlenbeck_policy():
    torch.manual_seed(42)

    mu = Regressor(LinearApproximator,  input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    w = torch.randn(pi.weights_size)
    pi.set_weights(w)
    assert torch.equal(pi.get_weights(), w)

    state = torch.randn(5)

    policy_state = pi.reset()

    action, policy_state = pi.draw_action(state, policy_state)
    action_test = torch.tensor([-1.95896171,  1.91292747])
    assert torch.allclose(action, action_test)

    policy_state = pi.reset()
    action, policy_state = pi.draw_action(state, policy_state)
    action_test = torch.tensor([-1.94161061,  1.92233358])
    assert torch.allclose(action, action_test)

    try:
        pi(state, action)
    except NotImplementedError:
        pass
    else:
        assert False

# TODO Missing test for clipped gaussian!

