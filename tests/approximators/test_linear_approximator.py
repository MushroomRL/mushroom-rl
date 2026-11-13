import numpy as np
import pytest

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis


def test_linear():
    np.random.seed(1)

    a = np.random.rand(1000, 3)

    k = np.random.rand(3, 2)
    b = a.dot(k) + np.random.randn(1000, 2)

    approximator = LinearApproximator(input_shape=(3,), output_shape=(2,))

    approximator.fit(a, b)

    x = np.random.rand(2, 3)
    y = approximator.predict(x)
    y_test = np.load('tests/approximators/data/test_linear_y_0.npy')

    assert np.allclose(y, y_test)
    assert approximator.predict(x[:1]).shape == (1, 2) and approximator.predict(x[0]).shape == (2,)

    point = np.random.randn(3,)
    derivative = approximator.diff(point)

    lp = len(point)
    for i in range(derivative.shape[1]):
        assert (derivative[i * lp:(i + 1) * lp, i] == point).all()

    for output_shape in [(), (1,)]:
        assert LinearApproximator(input_shape=(3,), output_shape=output_shape).diff(point).shape == (3, *output_shape)

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new) and random_weight_new.dtype == np.float64
    assert not np.any(np.equal(random_weights, old_weights))

    random_weights += 1.

    assert np.array_equal(approximator.get_weights(), random_weight_new)

    np.random.seed(1)

    phi = Features(PolynomialBasis.generate(2, 2))
    approximator = LinearApproximator(input_shape=(2,), output_shape=(3,), phi=phi)

    assert approximator.input_shape == (2,)
    assert approximator.output_shape == (3,)
    assert approximator.weights_size == 3 * phi.size

    x = np.random.rand(1000, 2)
    k = np.random.rand(2, 3)
    y = x.dot(k) + np.random.randn(1000, 3) * 0.1

    approximator.fit(x, y)

    x_test = np.random.rand(2, 2)
    y_hat = approximator.predict(x_test)
    y_test = np.load('tests/approximators/data/test_linear_y_1.npy')

    assert np.allclose(y_hat, y_test)

    derivative = approximator.diff(x_test[0])

    assert derivative.shape == (approximator.weights_size, 3)
    assert np.array_equal(derivative[:phi.size, 0], phi(x_test[0]))
    assert not np.any(derivative[phi.size:, 0])

    with pytest.raises(TypeError):
        LinearApproximator()

    with pytest.raises(AssertionError):
        LinearApproximator(input_shape=[(2,), (3,)], output_shape=(1,))

    with pytest.raises(AssertionError):
        LinearApproximator(input_shape=(2, 3), output_shape=(1,))
