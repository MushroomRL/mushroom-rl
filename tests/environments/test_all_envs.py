import numpy as np

from mushroom.environments.atari import Atari
from mushroom.environments.car_on_hill import CarOnHill
from mushroom.environments.cart_pole import CartPole
from mushroom.environments.generators import generate_grid_world,\
    generate_simple_chain, generate_taxi
from mushroom.environments.grid_world import GridWorld, GridWorldVanHasselt
from mushroom.environments.gym_env import Gym
from mushroom.environments.inverted_pendulum import InvertedPendulum
from mushroom.environments.lqr import LQR
from mushroom.environments.puddle_world import PuddleWorld
from mushroom.environments.segway import Segway
from mushroom.environments.ship_steering import ShipSteering

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


def test_atari():
    np.random.seed(1)
    mdp = Atari(name='PongDeterministic-v4')
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.load('tests/environments/test_atari_1.npy')

    assert np.allclose(ns, ns_test)

    mdp = Atari(name='PongNoFrameskip-v4')
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.load('tests/environments/test_atari_2.npy')

    assert np.allclose(ns, ns_test)


def test_car_on_hill():
    np.random.seed(1)
    mdp = CarOnHill()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.array([-0.29638141, -0.05527507])
    mdp.render()

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
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.array([1.5195833, -2.82335548])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_finite_mdp():
    np.random.seed(1)
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 4


def test_grid_world():
    np.random.seed(1)
    mdp = GridWorld(start=(0, 0), goal=(2, 2), height=3, width=3)
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    mdp.render()

    assert ns == 0

    np.random.seed(1)
    mdp = GridWorldVanHasselt()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 6

    np.random.seed(5)
    mdp = generate_grid_world('tests/environments/grid.txt', .9, 1, -1)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 4


def test_gym():
    np.random.seed(1)
    mdp = Gym('Acrobot-v1', 1000, .99)
    mdp.seed(1)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.array([0.99989477, 0.01450661, 0.97517825, -0.22142128,
                        -0.02323116, 0.40630765])

    assert np.allclose(ns, ns_test)


def test_inverted_pendulum():
    np.random.seed(1)
    mdp = InvertedPendulum()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand(mdp.info.action_space.shape[0])])
    ns_test = np.array([1.62134054, 1.0107062])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_lqr():
    np.random.seed(1)
    mdp = LQR.generate(2)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step(np.random.rand(mdp.info.action_space.shape[0]))
    ns_test = np.array([12.35564605, 14.98996889])

    assert np.allclose(ns, ns_test)


def test_puddle_world():
    np.random.seed(1)
    mdp = PuddleWorld()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.array([0.41899424, 0.4022506])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_segway():
    np.random.seed(1)
    mdp = Segway()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array([-0.64112019, -4.92869367, 10.33970413])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_ship_steering():
    np.random.seed(1)
    mdp = ShipSteering()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array([0., 7.19403055, 1.66804923, 0.08134399])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_taxi():
    np.random.seed(1)
    mdp = generate_taxi('tests/environments/taxi.txt')
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = 5

    assert np.allclose(ns, ns_test)
