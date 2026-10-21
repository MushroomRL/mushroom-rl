import numpy as np
import pytest
from mushroom_rl.environments.gymnasium_env import Gymnasium
from mushroom_rl.environments.atari import Atari
from mushroom_rl.environments.minigrid_env import MiniGrid, MiniGridRGB
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.environments.cart_pole import CartPole
from mushroom_rl.environments.grid_world import GridWorld
from mushroom_rl.environments.grid_world_van_hasselt import GridWorldVanHasselt
from mushroom_rl.environments.simple_chain import SimpleChain
from mushroom_rl.environments.taxi import Taxi
from mushroom_rl.environments.inverted_pendulum import InvertedPendulum
from mushroom_rl.environments.lqr import LQR
from mushroom_rl.environments.puddle_world import PuddleWorld
from mushroom_rl.environments.segway import Segway
from mushroom_rl.environments.ship_steering import ShipSteering

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


def _assert_step_returns_not_mutated(states, snapshots):
    for state, snapshot in zip(states, snapshots):
        assert np.allclose(state, snapshot)


def test_atari():
    np.random.seed(1)
    mdp = Atari(name='ALE/Pong-v5')
    mdp.seed(1)
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)

    ns_test = np.load('tests/environments/test_atari_1.npy')

    assert np.allclose(ns, ns_test)


def test_car_on_hill():
    np.random.seed(1)
    mdp = CarOnHill()
    mdp.reset()
    mdp.render()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
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
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([1.25234221, 2.44501606])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_finite_mdp():
    np.random.seed(1)
    mdp = SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1, gamma=.9)
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    mdp.render()

    assert ns == 4


def test_grid_world():
    np.random.seed(1)
    mdp = GridWorld.from_size(start=(0, 0), goal=(2, 2), height=3, width=3)
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    mdp.render()

    assert ns == 3

    np.random.seed(1)
    mdp = GridWorldVanHasselt()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 3

    np.random.seed(5)
    mdp = GridWorld.from_file('tests/environments/grid.txt', prob=.9)
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])

    assert ns == 4


def test_gymnasium():
    np.random.seed(1)
    mdp = Gymnasium('Acrobot-v1', 1000, .99)
    mdp.seed(1)
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([0.9996687, -0.02573896, 0.9839331, -0.17853762, -0.17821608, 0.5534913])

    assert np.allclose(ns, ns_test)


def test_gymnasium_reset_state():
    mdp = Gymnasium('MountainCar-v0', 1000, .99)
    mdp.seed(1)
    initial_state = np.array([-.75, -.02])
    state, _ = mdp.reset(initial_state)
    assert np.array_equal(state, initial_state)
    assert not np.shares_memory(state, initial_state)
    for i in range(5):
        ns, r, ab, _ = mdp.step(np.array([2]))
    ns_test = np.array([-0.8097278, -0.0063345535])

    assert np.allclose(ns, ns_test)
    assert r == -1.
    assert not ab
    assert np.array_equal(initial_state, np.array([-.75, -.02]))

    mdp = Gymnasium('CartPole-v1', 1000, .99)
    mdp.seed(1)
    initial_state = np.array([.1, 0., .1, 0.])
    state, _ = mdp.reset(initial_state)
    assert np.array_equal(state, initial_state)
    for i in range(5):
        ns, r, ab, _ = mdp.step(np.array([1]))
    ns_test = np.array([0.13872232, 0.96879756, 0.04791078, -1.3157222])

    assert np.allclose(ns, ns_test)
    assert r == 1.
    assert not ab


def test_gymnasium_reset_state_not_supported():
    for name in ['Pendulum-v1', 'FrozenLake-v1']:
        mdp = Gymnasium(name, 100, .99)
        state, _ = mdp.reset()
        with pytest.raises(NotImplementedError):
            mdp.reset(state)


def test_gymnasium_render():
    mdp = Gymnasium('Pendulum-v1', 200, .99, headless=True)
    for _ in range(2):
        mdp.seed(1)
        mdp.reset()
        frames = []
        for i in range(3):
            mdp.step(np.array([.5]))
            frames.append(mdp.render(record=True))
            assert np.array_equal(frames[-1], mdp.env.render())
            assert mdp.render(record=False) is None
        mdp.stop()

        assert all(frame.shape == (500, 500, 3) and frame.dtype == np.uint8 for frame in frames)
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])


def test_inverted_pendulum():
    np.random.seed(1)
    mdp = InvertedPendulum()
    mdp.reset()
    mdp.render()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand(mdp.info.action_space.shape[0])])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([1.62134054, 1.0107062])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_lqr():
    np.random.seed(1)
    mdp = LQR.generate(2)
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step(np.random.rand(mdp.info.action_space.shape[0]))
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([12.35564605, 14.98996889])

    assert np.allclose(ns, ns_test)

    A = np.eye(3)
    B = np.array([[2 / 3, 0], [1 / 3, 1 / 3], [0, 2 / 3]])
    Q = np.array([[0.1, 0., 0.], [0., 0.9, 0.], [0., 0., 0.1]])
    R = np.array([[0.1, 0.], [0., 0.9]])
    mdp = LQR(A, B, Q, R, max_pos=11.0, max_action=0.5, episodic=True)
    mdp.reset()

    states, snapshots = [], []

    a_test = np.array([1.0, 0.3])
    ns, r, ab, _ = mdp.step(a_test)
    states.append(ns)
    snapshots.append(np.copy(ns))
    ns_test = np.array([10.23333333, 10.16666667, 10.1])
    assert np.allclose(ns, ns_test) and np.allclose(r, -107.917) and not ab

    a_test = np.array([0.4, -0.1])
    ns, r, ab, _ = mdp.step(a_test)
    states.append(ns)
    snapshots.append(np.copy(ns))
    ns_test = np.array([10.5, 10.26666667, 10.03333333])
    assert np.allclose(ns, ns_test) and np.allclose(r, -113.72311111111117) and not ab

    a_test = np.array([0.5, 0.6])
    ns, r, ab, _ = mdp.step(a_test)
    states.append(ns)
    snapshots.append(np.copy(ns))
    ns_test = np.array([10.83333333, 10.6, 10.36666667])
    assert np.allclose(ns, ns_test) and np.allclose(r, -116.20577777777778) and not ab

    a_test = np.array([0.3, -0.7])
    ns, r, ab, _ = mdp.step(a_test)
    states.append(ns)
    snapshots.append(np.copy(ns))
    ns_test = np.array([11.03333333, 10.53333333, 10.03333333])
    assert np.allclose(ns, ns_test) and np.allclose(r, -1210.0) and ab

    _assert_step_returns_not_mutated(states, snapshots)


def test_puddle_world():
    np.random.seed(1)
    mdp = PuddleWorld()
    mdp.reset()
    mdp.render()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([0.41899424, 0.4022506])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_segway():
    np.random.seed(1)
    mdp = Segway()
    mdp.reset()
    mdp.render()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([-0.64112019, -4.92869367, 10.33970413])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_ship_steering():
    np.random.seed(1)
    mdp = ShipSteering()
    mdp.reset()
    mdp.render()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)
    ns_test = np.array([0., 7.19403055, 1.66804923, 0.08134399])
    mdp.render()

    assert np.allclose(ns, ns_test)


def test_taxi():
    np.random.seed(1)
    mdp = Taxi.generate()
    mdp.reset()
    mdp.render()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
    mdp.render()
    ns_test = 4

    assert np.allclose(ns, ns_test)


def test_minigrid():
    np.random.seed(1)
    mdp = MiniGrid('MiniGrid-Empty-5x5-v0', fixed_seed=1)
    assert mdp.info.observation_space.data_type == np.uint8
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)

    ns_test = np.load('tests/environments/test_minigrid_1.npy')

    assert np.allclose(ns, ns_test)


def test_minigrid_rgb():
    np.random.seed(1)
    mdp = MiniGridRGB('MiniGrid-Empty-5x5-v0', fixed_seed=1)
    assert mdp.info.observation_space.data_type == np.uint8
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        states.append(ns)
        snapshots.append(np.copy(ns))
    _assert_step_returns_not_mutated(states, snapshots)

    ns_test = np.load('tests/environments/test_minigrid_2.npy')

    assert np.allclose(ns, ns_test)


def test_minigrid_seed():
    mdp = MiniGrid('MiniGrid-DoorKey-5x5-v0')
    mdp.seed(5)
    first_run = [mdp.reset()[0] for _ in range(3)]

    mdp = MiniGrid('MiniGrid-DoorKey-5x5-v0')
    mdp.seed(5)
    second_run = [mdp.reset()[0] for _ in range(3)]

    mdp = MiniGrid('MiniGrid-DoorKey-5x5-v0')
    mdp.seed(6)
    other_run = [mdp.reset()[0] for _ in range(3)]

    for first, second in zip(first_run, second_run):
        assert np.array_equal(first, second)

    assert not all(np.array_equal(first, other) for first, other in zip(first_run, other_run))


def test_minigrid_fixed_seed_overrides_seed():
    mdp = MiniGrid('MiniGrid-DoorKey-5x5-v0', fixed_seed=5)
    mdp.seed(6)
    states = [mdp.reset()[0] for _ in range(3)]

    for state in states[1:]:
        assert np.array_equal(states[0], state)
