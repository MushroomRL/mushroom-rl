import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.grid_world import GridWorld
from mushroom_rl.environments.grid_world_van_hasselt import GridWorldVanHasselt
from mushroom_rl.environments.simple_chain import SimpleChain
from mushroom_rl.environments.taxi import Taxi
from mushroom_rl.solvers.dynamic_programming import value_iteration, policy_iteration
from mushroom_rl.utils.viewer import Viewer


def test_from_size_matches_from_file(tmp_path):
    path = tmp_path / 'three_by_three.txt'
    path.write_text('S..\n'
                    '...\n'
                    '..G\n')

    from_file = GridWorld.from_file(str(path))
    from_size = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    assert np.array_equal(from_file.grid_map, from_size.grid_map)
    assert np.array_equal(from_file.cell_list, from_size.cell_list)
    assert np.allclose(from_file.p, from_size.p)
    assert np.allclose(from_file.r, from_size.r)
    assert np.allclose(from_file.mu, from_size.mu)


def test_missing_trailing_newline(tmp_path):
    path = tmp_path / 'no_newline.txt'
    path.write_text('S..\n'
                    '...\n'
                    '..G')

    mdp = GridWorld.from_file(str(path))

    assert mdp.grid_map.shape == (1, 3, 3)
    assert mdp.info.observation_space.n == 9


def test_walls_and_holes():
    mdp = GridWorld.from_file('tests/environments/grid.txt', prob=.9)

    assert mdp.info.observation_space.n == 11
    assert mdp.grid_map.shape == (1, 6, 5)

    hole = np.argwhere((mdp.cell_list == [0, 1, 3]).all(axis=1)).item()
    goal = np.argwhere((mdp.cell_list == [0, 4, 1]).all(axis=1)).item()

    assert not np.any(mdp.p[hole])
    assert not np.any(mdp.p[goal])
    assert np.allclose(mdp.p[0, 0, 0], 1.)


def test_unknown_symbol(tmp_path):
    path = tmp_path / 'unknown.txt'
    path.write_text('S.?\n'
                    '..G\n')

    with pytest.raises(ValueError):
        GridWorld.from_file(str(path))


def test_missing_goal(tmp_path):
    path = tmp_path / 'no_goal.txt'
    path.write_text('S..\n'
                    '...\n')

    with pytest.raises(AssertionError):
        GridWorld.from_file(str(path))


def test_missing_start(tmp_path):
    path = tmp_path / 'no_start.txt'
    path.write_text('...\n'
                    '..G\n')

    with pytest.raises(AssertionError):
        GridWorld.from_file(str(path))


def test_dynamic_programming():
    mdp = GridWorld.from_file('tests/environments/grid.txt', prob=.9)

    value_vi = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-10)
    value_pi, policy_pi = policy_iteration(mdp.p, mdp.r, mdp.info.gamma)

    value_test = np.array([0.62083419, 0.69748038, 0., 0.69748038, 0.78358907,
                           0.69748038, 0.88032846, 0.78358907, 0., 0.98901099, 0.88032846])

    assert np.allclose(value_vi, value_test)
    assert np.allclose(value_pi, value_test)
    assert np.array_equal(policy_pi, np.array([1, 1, 0, 3, 1, 1, 1, 1, 0, 2, 2]))


def test_van_hasselt():
    np.random.seed(3)
    mdp = GridWorldVanHasselt()

    assert mdp.info.observation_space.n == 10
    assert len(mdp.cell_list) == mdp.info.observation_space.n
    assert mdp._get_state_id(np.array([0, 0, 2])) == 2
    assert np.allclose(mdp.r[0, 0, 0], -1.)
    assert np.allclose(mdp.r[2, 0, 9], 5.)
    assert np.allclose(mdp.p[2, 0, 9], 1.)
    assert not np.any(mdp.p[9])

    value = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-12)
    gamma = mdp.info.gamma

    assert np.allclose(value[np.argmax(mdp.mu)], 5 * gamma ** 4 - sum(gamma ** k for k in range(4)))

    rewards = []
    mdp.reset(np.array([6]))
    for i in range(50):
        _, reward, absorbing, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        rewards.append(reward)
        if absorbing:
            break

    assert set(rewards[:-1]).issubset({-12., 10.})
    assert absorbing and rewards[-1] == 5.


def test_taxi_passengers():
    mdp = Taxi.from_file('tests/environments/taxi.txt')

    assert mdp.grid_map.shape == (8, 6, 7)
    assert mdp.info.observation_space.n == 252

    start = np.argmax(mdp.mu)
    passenger = np.argwhere((mdp.cell_list == [1, 0, 2]).all(axis=1)).item()
    below_passenger = np.argwhere((mdp.cell_list == [0, 1, 2]).all(axis=1)).item()

    assert np.allclose(mdp.cell_list[start], [0, 0, 0])
    assert np.allclose(mdp.p[below_passenger, 0, passenger], .9)
    assert np.allclose(mdp.p.sum(axis=2)[np.any(mdp.p, axis=(1, 2))], 1.)
    assert np.array_equal(np.unique(mdp.r), np.array([0., 1., 3., 15.]))


def test_taxi_from_size():
    mdp = Taxi.from_size(height=3, width=3, goal=(2, 2), passengers=((0, 2), (2, 0)), goal_rewards=(0, 1, 5))

    assert mdp.grid_map.shape == (4, 3, 3)
    assert mdp.info.observation_space.n == 32
    assert np.array_equal(np.unique(mdp.grid_map), np.array(['.', 'G', 'P', 'S']))
    assert np.array_equal(np.unique(mdp.r), np.array([0., 1., 5.]))
    assert np.allclose(mdp.p.sum(axis=2)[np.any(mdp.p, axis=(1, 2))], 1.)


def test_taxi_passenger_on_occupied_cell():
    with pytest.raises(AssertionError):
        Taxi.from_size(height=3, width=3, goal=(2, 2), passengers=((0, 0),), goal_rewards=(0, 1))


def test_taxi_wrong_rewards():
    with pytest.raises(AssertionError):
        Taxi.from_file('tests/environments/taxi.txt', goal_rewards=(0, 1))


def test_simple_chain():
    mdp = SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1, gamma=.9)

    assert mdp.info.observation_space.n == 5
    assert mdp.info.action_space.n == 2
    assert np.allclose(mdp.p.sum(axis=2), 1.)
    assert np.allclose(mdp.p[0, 1, 0], 1.)
    assert np.allclose(mdp.p[4, 0, 4], 1.)
    assert np.allclose(mdp.r[1, 0, 2], 1.)
    assert np.allclose(mdp.r[2, 0, 2], 0.)


def test_viewer_window_size():
    square = Viewer(10, 10, min_scale=40)
    corridor = Viewer(2, 20, min_scale=40)
    wide = Viewer(48, 1, min_scale=40)

    assert square.size == (500, 500)
    assert corridor.size == (500, 1080)
    assert wide.size == (1920, 100)

    for viewer in (square, corridor, wide):
        assert viewer.fits
        assert viewer.size[0] >= 500 and viewer.size[1] >= 100


def test_viewer_does_not_fit_the_screen():
    assert not Viewer(60, 60, min_scale=40).fits
    assert not Viewer(100, 1, min_scale=40).fits
    assert not Viewer(48, 2084, min_scale=40).fits


def test_viewer_margin_centres_the_grid():
    padded = Viewer(48, 1, min_scale=40)
    exact = Viewer(10, 10, min_scale=40)

    assert np.allclose(exact._margin, [0., 0.])
    assert np.allclose(padded._margin, [0., .75])


def test_cell_center():
    mdp = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    assert np.allclose(mdp._cell_center(0, 0), [.5, 2.5])
    assert np.allclose(mdp._cell_center(2, 2), [2.5, .5])


def test_simple_chain_wrapping():
    short = SimpleChain(5, [2], .8, 1)
    long = SimpleChain(600, [42, 300], .8, 1)

    assert short._n_rows == 1 and short._n_columns == 5
    assert short._viewer.size == (500, 100)
    assert short._cell_of(4) == (0, 4)

    assert long._n_rows == 13 and long._n_columns == 48
    assert long._viewer.size == (1920, 520)
    assert long._cell_of(0) == (0, 0)
    assert long._cell_of(72) == (1, 24)
    assert long._cell_of(599) == (12, 23)


def test_finite_mdp_default_viewer():
    short = FiniteMDP(np.zeros((6, 2, 6)), np.zeros((6, 2, 6)))
    wrapped = FiniteMDP(np.zeros((600, 2, 600)), np.zeros((600, 2, 600)))

    assert short._n_rows == 1 and short._n_columns == 6
    assert short._viewer.size == (500, 100)
    assert short._cell_of(5) == (0, 5)

    assert wrapped._n_rows == 13 and wrapped._n_columns == 48
    assert wrapped._viewer.size == (1920, 520)
    assert wrapped._cell_of(72) == (1, 24)
