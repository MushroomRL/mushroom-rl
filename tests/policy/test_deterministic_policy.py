from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.utils.numerical_gradient import numerical_diff_policy

import numpy as np


def test_deterministic_policy():
    np.random.seed(88)

    n_dims = 5

    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(2,))

    pi = DeterministicPolicy(approximator)

    w_new = np.random.rand(pi.weights_size)

    w_old = pi.get_weights()
    pi.set_weights(w_new)

    assert np.array_equal(w_new, approximator.get_weights())
    assert not np.array_equal(w_old, w_new)
    assert np.array_equal(w_new, pi.get_weights())

    s_test_1 = np.random.randn(5)
    s_test_2 = np.random.randn(5)
    a_test = approximator.predict(s_test_1)

    assert pi.get_regressor() == approximator

    assert pi(s_test_1, a_test) == 1
    assert pi(s_test_2, a_test) == 0

    a_stored = np.array([-1.86941072, -0.1789696])
    action, _ = pi.draw_action(s_test_1)
    assert np.allclose(action, a_stored)

