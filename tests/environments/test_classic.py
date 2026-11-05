import numpy as np
import pytest

from helper.envs import box_action, discrete_action, rollout, scalar_action

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


@pytest.mark.parametrize('env, action, ns_test', [
    pytest.param(CartPole, discrete_action, [1.25234221, 2.44501606], id='cart_pole'),
    pytest.param(InvertedPendulum, box_action, [1.62134054, 1.0107062], id='inverted_pendulum'),
    pytest.param(PuddleWorld, discrete_action, [0.41899424, 0.4022506], id='puddle_world'),
    pytest.param(Segway, scalar_action, [-0.64112019, -4.92869367, 10.33970413], id='segway'),
    pytest.param(ShipSteering, scalar_action, [0., 7.19403055, 1.66804923, 0.08134399], id='ship_steering')
])
def test_rollout(env, action, ns_test):
    np.random.seed(1)
    ns = rollout(env(), action, render=True)

    assert np.allclose(ns, np.array(ns_test))


def test_car_on_hill():
    np.random.seed(1)
    mdp = CarOnHill()
    ns = rollout(mdp, discrete_action, render=True)
    ns_test = np.array([-0.29638141, -0.05527507])

    assert np.allclose(ns, ns_test)
    assert np.allclose(mdp._angle(ns_test[0]), -1.141676764064636)
    assert np.allclose(mdp._height(ns_test[0]), 0.9720652903871763)


def test_discrete_envs():
    np.random.seed(1)
    assert rollout(SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1, gamma=.9), discrete_action,
                   render=True) == 4

    np.random.seed(1)
    assert rollout(GridWorld.from_size(start=(0, 0), goal=(2, 2), height=3, width=3), discrete_action,
                   render=True) == 3

    np.random.seed(1)
    assert rollout(GridWorldVanHasselt(), discrete_action) == 3

    np.random.seed(5)
    assert rollout(GridWorld.from_file('tests/environments/data/grid.txt', prob=.9), discrete_action) == 4

    np.random.seed(1)
    assert np.allclose(rollout(Taxi.generate(), discrete_action, render=True), 4)


def test_lqr():
    np.random.seed(1)
    mdp = LQR.generate(2)
    ns = rollout(mdp, lambda mdp: np.random.rand(mdp.info.action_space.shape[0]))

    assert np.allclose(ns, np.array([12.35564605, 14.98996889]))

    A = np.eye(3)
    B = np.array([[2 / 3, 0], [1 / 3, 1 / 3], [0, 2 / 3]])
    Q = np.array([[0.1, 0., 0.], [0., 0.9, 0.], [0., 0., 0.1]])
    R = np.array([[0.1, 0.], [0., 0.9]])
    mdp = LQR(A, B, Q, R, max_pos=11.0, max_action=0.5, episodic=True)
    mdp.reset()

    states, snapshots = [], []
    for a_test, ns_test, r_test, ab_test in [([1.0, 0.3], [10.23333333, 10.16666667, 10.1], -107.917, False),
                                             ([0.4, -0.1], [10.5, 10.26666667, 10.03333333], -113.72311111111117,
                                              False),
                                             ([0.5, 0.6], [10.83333333, 10.6, 10.36666667], -116.20577777777778,
                                              False),
                                             ([0.3, -0.7], [11.03333333, 10.53333333, 10.03333333], -1210.0, True)]:
        ns, r, ab, _ = mdp.step(np.array(a_test))
        states.append(ns)
        snapshots.append(np.copy(ns))

        assert np.allclose(ns, np.array(ns_test)) and np.allclose(r, r_test) and ab == ab_test

    for state, snapshot in zip(states, snapshots):
        assert np.allclose(state, snapshot)

    mdp = LQR.generate(2, max_pos=1.)
    mdp.reset(np.array([.9, 0.]))
    ns, r, ab, _ = mdp.step(np.array([5., 0.]))

    assert np.array_equal(ns, [1., 0.]) and np.isclose(r, -3.229) and not ab


def test_reset():
    for mdp, state, expected in [(ShipSteering(), [1., 2., .5, .1], [1., 2., .5, .1]),
                                 (Segway(), [3 * np.pi, .1, .2], [np.pi, .1, .2]),
                                 (InvertedPendulum(), [3 * np.pi, 100.], [np.pi, 2.5 * np.pi]),
                                 (PuddleWorld(), [.3, .4], [.3, .4]),
                                 (LQR.generate(2), [1., 2.], [1., 2.]),
                                 (LQR.generate(2, initial_state=np.array([.5, -.5])), None, [.5, -.5])]:
        assert np.allclose(mdp.reset(None if state is None else np.array(state))[0], expected)

    np.random.seed(1)
    assert np.allclose(ShipSteering(small=False).reset()[0], np.load('tests/environments/data/test_reset.npy'))
    np.random.seed(1)
    assert np.allclose(Segway(random_start=True).reset()[0], [-0.26068306, 0., 0.])
    np.random.seed(1)
    assert np.allclose(InvertedPendulum(random_start=True).reset()[0], [-0.52136612, 0.])
    np.random.seed(1)
    assert np.allclose(LQR.generate(2, random_init=True).reset()[0], [-0.49786797, 1.32194696])


def test_goals():
    mdp = ShipSteering()
    mdp.reset(np.array([109., 109., np.pi / 4, 0.]))
    ns, r, ab, _ = mdp.step(np.array([0.]))

    assert np.allclose(ns, [110.27279221, 110.27279221, np.pi / 4, 0.]) and r == 0 and ab
    assert not mdp._through_gate(np.array([0., 0.]), np.array([20., -20.]))

    mdp.stop()

    np.random.seed(1)
    mdp = PuddleWorld(reward_goal=5.)
    mdp.reset(np.array([1., 1.]))
    ns, r, ab, _ = mdp.step([0])

    assert np.allclose(ns, [0.9458511, 1.]) and r == 5. and ab

    mdp.stop()
