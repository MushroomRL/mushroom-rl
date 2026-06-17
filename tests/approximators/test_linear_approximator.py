import numpy as np

from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import LinearApproximator


def test_linear_approximator():
    np.random.seed(1)

    a = np.random.rand(1000, 3)

    k = np.random.rand(3, 2)
    b = a.dot(k) + np.random.randn(1000, 2)

    approximator = LinearApproximator(input_shape=(3,), output_shape=(2,))

    approximator.fit(a, b)

    x = np.random.rand(2, 3)
    y = approximator.predict(x)
    y_test = np.array([[0.57638247, 0.1573216],
                       [0.11388247, 0.24123678]])

    assert np.allclose(y, y_test)

    point = np.random.randn(3,)
    derivative = approximator.diff(point)

    lp = len(point)
    for i in range(derivative.shape[1]):
        assert (derivative[i * lp:(i + 1) * lp, i] == point).all()

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))

    n_actions = 2
    s = np.random.rand(1000, 3)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    approximator = QApproximator(LinearApproximator, n_actions=n_actions, input_shape=(3,))

    approximator.fit(s, a, q)

    x_s = np.random.rand(2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))
    y = approximator.predict(x_s, x_a)
    y_test = np.array([0.49225698, 0.69660881])
    assert np.allclose(y, y_test)

    y = approximator.predict(x_s)
    y_test = np.array([[0.49225698, 0.44154141],
                       [0.69660881, 0.69060195]])
    assert np.allclose(y, y_test)

    approximator = QApproximator(LinearApproximator, n_actions=n_actions, input_shape=(3,))

    approximator.fit(s, a, q)

    gradient = approximator.diff(x_s[0], x_a[0])
    gradient_test = np.array([0.88471362, 0.11666548, 0.45466254, 0., 0., 0.])

    assert np.allclose(gradient, gradient_test)
