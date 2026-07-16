import numpy as np

from mushroom_rl.approximators.parametric import CMAC
from mushroom_rl.features.tiles import Tiles


def test_cmac_approximator():
    np.random.seed(1)

    x = np.random.rand(1000, 2)

    k1 = np.random.rand(2)
    k2 = np.random.rand(2)

    y = np.array([np.sin(x.dot(k1)*2*np.pi), np.sin(x.dot(k2)*2*np.pi)]).T

    tilings = Tiles.generate(10, [10, 10], np.zeros(2), np.ones(2))
    approximator = CMAC(tilings=tilings, input_shape=(2,), output_shape=(2,))

    approximator.fit(x, y)

    x = np.random.rand(2, 2)
    y_hat = approximator.predict(x)

    y_test = np.array([[-0.73581504,  0.90877225],
                       [-0.95854488, -0.72429239]])

    assert np.allclose(y_hat, y_test)

    point = np.random.rand(2)
    derivative = approximator.diff(point)

    assert np.array_equal(np.sum(derivative, axis=0), np.ones(2)*10)
    assert len(derivative) == approximator.weights_size

    phi_point = approximator._phi(point)
    n_phi = phi_point.size

    assert np.array_equal(derivative[:n_phi, 0], phi_point)
    assert np.array_equal(derivative[n_phi:, 1], phi_point)
    assert not np.any(derivative[n_phi:, 0])
    assert not np.any(derivative[:n_phi, 1])

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))


def test_cmac_approximator_outside():
    np.random.seed(1)

    x = np.random.rand(1000, 2)
    k = np.random.rand(2)
    y = np.sin(x.dot(k) * 2 * np.pi)

    tilings = Tiles.generate(10, [10, 10], np.zeros(2), np.ones(2))
    approximator = CMAC(tilings=tilings, input_shape=(2,))
    approximator.fit(x, y)

    y_hat = approximator.predict(np.array([[.5, .5], [10., 10.], [-3., .5]]))

    assert not np.allclose(y_hat[0], 0.)
    assert np.array_equal(y_hat[1:], np.zeros(2))
