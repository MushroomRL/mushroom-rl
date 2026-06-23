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
    approximator = CMAC(tilings=tilings, output_shape=(2,))

    approximator.fit(x, y)

    x = np.random.rand(2, 2)
    y_hat = approximator.predict(x)
    y_true = np.array([np.sin(x.dot(k1) * 2 * np.pi), np.sin(x.dot(k2) * 2 * np.pi)]).T

    y_test = np.array([[-0.73581504,  0.90877225],
                       [-0.95854488, -0.72429239]])

    assert np.allclose(y_hat, y_test)

    point = np.random.rand(2)
    derivative = approximator.diff(point)

    assert np.array_equal(np.sum(derivative, axis=0), np.ones(2)*10)
    assert len(derivative) == approximator.weights_size

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))
