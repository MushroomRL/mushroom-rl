import numpy as np

from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric import *
from mushroom_rl.features.tiles import Tiles


def test_cmac_approximator():
    np.random.seed(1)

    # Generic regressor
    x = np.random.rand(1000, 2)

    k1 = np.random.rand(2)
    k2 = np.random.rand(2)

    y = np.array([np.sin(x.dot(k1)*2*np.pi), np.sin(x.dot(k2)*2*np.pi)]).T

    tilings = Tiles.generate(10, [10, 10], np.zeros(2), np.ones(2))
    approximator = Regressor(CMAC,
                             tilings=tilings,
                             input_shape=(2,),
                             output_shape=(2,))

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

    # Action regressor + Ensemble
    n_actions = 2
    s = np.random.rand(1000, 3)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    tilings = Tiles.generate(10, [10, 10, 10], np.zeros(3), np.ones(3))
    approximator = Regressor(CMAC,
                             tilings=tilings,
                             input_shape=(3,),
                             n_actions=n_actions,
                             n_models=5)

    approximator.fit(s, a, q)
    np.random.seed(2)
    x_s = np.random.rand(2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))
    y = approximator.predict(x_s, x_a, prediction='mean')
    y_test = np.array([[0.56235045, 0.25080909]])
    assert np.allclose(y, y_test)

    y = approximator.predict(x_s, x_a, prediction='sum')
    y_test = np.array([2.81175226, 1.25404543])
    assert np.allclose(y, y_test)

    y = approximator.predict(x_s, x_a, prediction='min')
    y_test = np.array([0.56235045, 0.25080909])
    assert np.allclose(y, y_test)

    y = approximator.predict(x_s)
    y_test = np.array([[0.10367145, 0.56235045],
                       [0.05575822, 0.25080909]])
    assert np.allclose(y, y_test)
