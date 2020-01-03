import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy


def test_ornstein_uhlenbeck_policy():
    np.random.seed(88)

    mu = Regressor(LinearApproximator,  input_shape=(5,), output_shape=(2,))
    pi = OrnsteinUhlenbeckPolicy(mu, sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    w = np.random.randn(pi.weights_size)
    pi.set_weights(w)
    assert np.array_equal(pi.get_weights(), w)

    state = np.random.randn(5)

    action = pi.draw_action(state)
    action_test = np.array([-1.95896171,  1.91292747])
    assert np.allclose(action, action_test)

    pi.reset()
    action = pi.draw_action(state)
    action_test = np.array([-1.94161061,  1.92233358])
    assert np.allclose(action, action_test)

    try:
        pi(state, action)
    except NotImplementedError:
        pass
    else:
        assert False

