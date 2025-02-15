import numpy as np
from mushroom_rl.environments.gymnasium_env import Gymnasium
from mushroom_rl.environments.atari import Atari
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.environments.cart_pole import CartPole
from mushroom_rl.environments.generators import generate_grid_world,\
    generate_simple_chain, generate_taxi
from mushroom_rl.environments.grid_world import GridWorld, GridWorldVanHasselt
from mushroom_rl.environments.inverted_pendulum import InvertedPendulum
from mushroom_rl.environments.lqr import LQR
from mushroom_rl.environments.puddle_world import PuddleWorld
from mushroom_rl.environments.segway import Segway
from mushroom_rl.environments.ship_steering import ShipSteering

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


def test_atari():
    np.random.seed(1)
    mdp = Atari(name='ALE/Pong-v5')
    mdp.seed(1)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    ns_test = np.load('tests/environments/test_atari_1.npy')

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
    ns_test = np.array([1.25234221, 2.44501606])
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

    assert ns == 5

    np.random.seed(1)
    mdp = GridWorldVanHasselt()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 2

    np.random.seed(5)
    mdp = generate_grid_world('tests/environments/grid.txt', .9, 1, -1)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 4


def test_gymnasium():
    np.random.seed(1)
    mdp = Gymnasium('Acrobot-v1', 1000, .99)
    mdp.seed(1)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    ns_test = np.array([0.9996687, -0.02573896,  0.9839331 , -0.17853762, -0.17821608,0.5534913])

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

    A = np.eye(3)
    B = np.array([[2 / 3, 0], [1 / 3, 1 / 3], [0, 2 / 3]])
    Q = np.array([[0.1, 0., 0.], [0., 0.9, 0.], [0., 0., 0.1]])
    R = np.array([[0.1, 0.], [0., 0.9]])
    mdp = LQR(A, B, Q, R, max_pos=11.0, max_action=0.5, episodic=True)
    mdp.reset()

    a_test = np.array([1.0, 0.3])
    ns, r, ab, _ = mdp.step(a_test)
    ns_test = np.array([10.23333333, 10.16666667, 10.1])
    assert np.allclose(ns, ns_test) and np.allclose(r, -107.917) and not ab

    a_test = np.array([0.4, -0.1])
    ns, r, ab, _ = mdp.step(a_test)
    ns_test = np.array([10.5, 10.26666667, 10.03333333])
    assert np.allclose(ns, ns_test) and np.allclose(r, -113.72311111111117) and not ab

    a_test = np.array([0.5, 0.6])
    ns, r, ab, _ = mdp.step(a_test)
    ns_test = np.array([10.83333333, 10.6, 10.36666667])
    assert np.allclose(ns, ns_test) and np.allclose(r, -116.20577777777778) and not ab

    a_test = np.array([0.3, -0.7])
    ns, r, ab, _ = mdp.step(a_test)
    ns_test = np.array([11.03333333, 10.53333333, 10.03333333])
    assert np.allclose(ns, ns_test) and np.allclose(r, -1210.0) and ab


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
