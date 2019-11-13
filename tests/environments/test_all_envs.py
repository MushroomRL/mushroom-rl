import numpy as np
from mushroom.environments.car_on_hill import CarOnHill
from mushroom.environments.cart_pole import CartPole
from mushroom.environments.inverted_pendulum import InvertedPendulum
from mushroom.environments.lqr import LQR
from mushroom.environments.segway import Segway
from mushroom.environments.ship_steering import ShipSteering


def test_car_on_hill():
    np.random.seed(1)
    mdp = CarOnHill()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(2)])
    ns_test = np.array([-0.29638141, -0.05527507])

    angle = mdp._angle(ns_test[0])
    angle_test = -1.141676764064636

    height = mdp._height(ns_test[0])
    height_test = 0.9720652903871763

    assert np.allclose(ns, ns_test)
    assert np.allclose(angle, angle_test)
    assert np.allclose(height, height_test)


def test_cartpole():
    np.random.seed(1)
    mdp = CartPole()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(3)])
    ns_test = np.array([1.5195833, -2.82335548])

    assert np.allclose(ns, ns_test)


def test_inverted_pendulum():
    np.random.seed(1)
    mdp = InvertedPendulum()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array([1.62134054, 1.0107062])

    assert np.allclose(ns, ns_test)


def test_lqr():
    np.random.seed(1)
    mdp = LQR.generate(2)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step(np.random.rand(2))
    ns_test = np.array([12.35564605, 14.98996889])

    assert np.allclose(ns, ns_test)


def test_segway():
    np.random.seed(1)
    mdp = Segway()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array([-0.64112019, -4.92869367, 10.33970413])

    assert np.allclose(ns, ns_test)


def test_ship_steering():
    np.random.seed(1)
    mdp = ShipSteering()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array([0., 7.19403055, 1.66804923, 0.08134399])

    assert np.allclose(ns, ns_test)
