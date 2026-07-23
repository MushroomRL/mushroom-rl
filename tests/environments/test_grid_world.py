import os

import numpy as np
import pytest

from mushroom_rl.core import Environment
from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.grid_world import GridWorld
from mushroom_rl.environments.grid_world_van_hasselt import GridWorldVanHasselt
from mushroom_rl.environments.simple_chain import SimpleChain
from mushroom_rl.environments.taxi import Taxi
from mushroom_rl.solvers.dynamic_programming import value_iteration, policy_iteration
from mushroom_rl.utils.viewer import Viewer

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def test_from_size_matches_from_file(tmp_path):
    path = tmp_path / 'three_by_three.txt'
    path.write_text('S..\n'
                    '...\n'
                    '..G\n')

    from_file = GridWorld.from_file(str(path))
    from_size = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    assert np.array_equal(from_file.grid_map, from_size.grid_map)
    assert np.array_equal(from_file.cell_list, from_size.cell_list)
    assert np.array_equal(from_file.p, from_size.p)
    assert np.array_equal(from_file.r, from_size.r)
    assert np.array_equal(from_file.iota, from_size.iota)


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

    assert (mdp.p[hole, :, hole] == 1.).all()
    assert (mdp.p[goal, :, goal] == 1.).all()
    assert mdp.p[0, 0, 0] == 1.


def test_absorbing_states():
    mdp = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    states = np.arange(mdp.info.observation_space.n)
    absorbing = (mdp.p[states, :, states] == 1.).all(axis=1)

    assert (mdp.p.sum(axis=2) == 1.).all()
    assert np.array_equal(np.argwhere(absorbing).ravel(), np.array([8]))
    assert not np.any(mdp.r[8])

    mdp.reset(np.array([5]))
    _, reward, absorbing, _ = mdp.step(np.array([1]))

    assert absorbing and reward == 1.

    next_state, reward, absorbing, _ = mdp.step(np.array([0]))

    assert next_state.item() == 8 and absorbing and reward == 0.


def test_too_big_to_render():
    with pytest.warns(UserWarning):
        mdp = FiniteMDP(np.full((2, 1, 2), .5), np.zeros((2, 1, 2)), viewer_shape=(100, 100))

    mdp.reset()

    assert mdp.render() is None
    assert mdp.render(record=True) is None


def test_min_scale_below_one():
    with pytest.raises(AssertionError):
        Viewer(10, 10, min_scale=0)

    with pytest.raises(AssertionError):
        FiniteMDP(np.full((6, 2, 6), 1 / 6), np.zeros((6, 2, 6)), min_scale=0)


def test_transitions_not_summing_to_one():
    with pytest.raises(AssertionError):
        FiniteMDP(np.zeros((3, 2, 3)), np.zeros((3, 2, 3)))


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
    assert mdp.r[0, 0, 0] == -1.
    assert mdp.r[2, 0, 9] == 5.
    assert mdp.p[2, 0, 9] == 1.
    assert (mdp.p[9, :, 9] == 1.).all()
    assert not np.any(mdp.r[9])

    value = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-12)
    gamma = mdp.info.gamma

    assert np.allclose(value[np.argmax(mdp.iota)], 5 * gamma ** 4 - sum(gamma ** k for k in range(4)))

    rewards = []
    mdp.reset(np.array([6]))
    for i in range(50):
        _, reward, absorbing, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        rewards.append(reward)
        if absorbing:
            break

    assert set(rewards[:-1]).issubset({-12., 10.})
    assert absorbing and rewards[-1] == 5.


def test_van_hasselt_from_map(tmp_path):
    path = tmp_path / 'walls.txt'
    path.write_text('S#.\n'
                    '.#.\n'
                    '..G\n')

    mdp = GridWorldVanHasselt.from_file(str(path))

    assert mdp.info.observation_space.n == 8
    assert np.array_equal(mdp.cell_list[-1], [0, 2, 2])
    assert (mdp.p[mdp._get_state_id(np.array([0, 2, 2])), :, -1] == 1.).all()
    assert not np.any(mdp.r[-1])


def test_van_hasselt_hole(tmp_path):
    path = tmp_path / 'hole.txt'
    path.write_text('S.*\n'
                    '...\n'
                    '..G\n')

    with pytest.raises(AssertionError):
        GridWorldVanHasselt.from_file(str(path))


def test_van_hasselt_two_goals(tmp_path):
    path = tmp_path / 'two_goals.txt'
    path.write_text('S.G\n'
                    '...\n'
                    '..G\n')

    with pytest.raises(AssertionError):
        GridWorldVanHasselt.from_file(str(path))


def test_generate_derives_the_corners():
    grid_world = GridWorld.generate(height=4, width=5)
    van_hasselt = GridWorldVanHasselt.generate(height=4, width=5)

    assert np.array_equal(np.argwhere(grid_world.grid_map[0] == 'S'), [[0, 0]])
    assert np.array_equal(np.argwhere(grid_world.grid_map[0] == 'G'), [[3, 4]])

    assert np.array_equal(np.argwhere(van_hasselt.grid_map[0] == 'S'), [[3, 0]])
    assert np.array_equal(np.argwhere(van_hasselt.grid_map[0] == 'G'), [[0, 4]])


def test_taxi_passengers():
    mdp = Taxi.generate()

    assert mdp.grid_map.shape == (8, 6, 7)
    assert mdp.info.observation_space.n == 252

    start = np.argmax(mdp.iota)
    passenger = np.argwhere((mdp.cell_list == [1, 0, 2]).all(axis=1)).item()
    below_passenger = np.argwhere((mdp.cell_list == [0, 1, 2]).all(axis=1)).item()

    assert np.array_equal(mdp.cell_list[start], [0, 0, 0])
    assert mdp.p[below_passenger, 0, passenger] == .9
    assert (mdp.p.sum(axis=2) == 1.).all()
    assert np.array_equal(np.unique(mdp.r), np.array([0., 1., 3., 15.]))


def test_taxi_from_size():
    mdp = Taxi.from_size(height=3, width=3, goal=(2, 2), passengers=((0, 2), (2, 0)), goal_rewards=(0, 1, 5))

    assert mdp.grid_map.shape == (4, 3, 3)
    assert mdp.info.observation_space.n == 32
    assert np.array_equal(np.unique(mdp.grid_map), np.array(['.', 'G', 'P', 'S']))
    assert np.array_equal(np.unique(mdp.r), np.array([0., 1., 5.]))
    assert (mdp.p.sum(axis=2) == 1.).all()


def test_taxi_generate_derives_the_layout():
    mdp = Taxi.generate(height=4, width=5)

    assert np.array_equal(np.argwhere(mdp.grid_map[0] == 'S'), [[0, 0]])
    assert np.array_equal(np.argwhere(mdp.grid_map[0] == 'G'), [[3, 4]])
    assert np.array_equal(np.argwhere(mdp.grid_map[0] == 'P'), [[0, 4], [3, 0]])
    assert np.array_equal(np.unique(mdp.r), np.array([0., 1., 3.]))


def test_taxi_generate_needs_both_dimensions():
    with pytest.raises(AssertionError):
        Taxi.generate(height=4)


def test_taxi_passenger_on_occupied_cell():
    with pytest.raises(AssertionError):
        Taxi.from_size(height=3, width=3, goal=(2, 2), passengers=((0, 0),), goal_rewards=(0, 1))


def test_taxi_wrong_rewards():
    with pytest.raises(AssertionError):
        Taxi.generate(goal_rewards=(0, 1))


def test_simple_chain():
    mdp = SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1, gamma=.9)

    assert mdp.info.observation_space.n == 5
    assert mdp.info.action_space.n == 2
    assert (mdp.p.sum(axis=2) == 1.).all()
    assert mdp.p[0, 1, 0] == 1.
    assert mdp.p[4, 0, 4] == 1.
    assert mdp.r[1, 0, 2] == 1.
    assert mdp.r[2, 0, 2] == 0.


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


def test_background_image_covers_the_environment_not_the_window():
    image = np.full((64, 64, 3), 255.)

    padded = Viewer(2, 20, min_scale=40)
    padded.background_image(image)
    padded_columns = np.argwhere(padded.get_frame().any(-1))[:, 1]
    padded.close()

    exact = Viewer(10, 10, min_scale=40)
    exact.background_image(image)
    exact_columns = np.argwhere(exact.get_frame().any(-1))[:, 1]
    exact.close()

    assert padded_columns.min() == 196 and padded_columns.max() == 303
    assert exact_columns.min() == 0 and exact_columns.max() == 499


def test_grid_draws_every_border():
    viewer = Viewer(5, 5, min_scale=40)
    viewer.grid(5, 5)
    frame = viewer.get_frame()
    viewer.close()

    rows = [row for row in range(frame.shape[0]) if frame[row].all()]
    columns = [column for column in range(frame.shape[1]) if frame[:, column].all()]

    assert rows == [0, 100, 200, 300, 400, 499]
    assert columns == [0, 100, 200, 300, 400, 499]


def test_cell_of_skips_the_walls():
    mdp = GridWorld.from_file('tests/environments/grid.txt', prob=.9)

    assert mdp._cell_of(0) == (1, 1)
    assert mdp._cell_of(5) == (2, 3)
    assert mdp._cell_of(10) == (4, 3)

    for state in range(mdp.info.observation_space.n):
        assert mdp._cell_of(state) == tuple(mdp.cell_list[state, 1:])


def test_cell_center():
    mdp = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    assert np.allclose(mdp._cell_center(0, 0), [.5, 2.5])
    assert np.allclose(mdp._cell_center(2, 2), [2.5, .5])


def test_simple_chain_defaults():
    mdp = Environment.make('SimpleChain')
    reference = SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1.)

    assert np.array_equal(mdp.p, reference.p)
    assert np.array_equal(mdp.r, reference.r)
    assert mdp.info.observation_space.n == 5 and mdp.info.action_space.n == 2


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
    short = FiniteMDP(np.full((6, 2, 6), 1 / 6), np.zeros((6, 2, 6)))
    wrapped = FiniteMDP(np.full((600, 2, 600), 1 / 600), np.zeros((600, 2, 600)))

    assert short._n_rows == 1 and short._n_columns == 6
    assert short._viewer.size == (500, 100)
    assert short._cell_of(5) == (0, 5)

    assert wrapped._n_rows == 13 and wrapped._n_columns == 48
    assert wrapped._viewer.size == (1920, 520)
    assert wrapped._cell_of(72) == (1, 24)
