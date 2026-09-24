import numpy as np
import pytest
import torch

from mushroom_rl.core import Dataset, MDPInfo, AgentInfo
from mushroom_rl.core.dataset import DatasetInfo, VectorizedDataset
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.core.spaces import Box


def make_dataset(lasts, backend='numpy'):
    n = len(lasts)
    states = np.arange(n * 2, dtype=float).reshape(n, 2)
    actions = np.zeros((n, 1))
    rewards = np.ones(n)
    return Dataset.from_array(states, actions, rewards, states + 1, np.zeros(n, dtype=bool),
                              np.array(lasts, dtype=bool), backend=backend, gamma=0.9)


def make_sequential_info():
    return DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None)


def make_grid_info(n_envs):
    return DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None, n_envs=n_envs)


def append_rows(dataset, lasts):
    for i, last in enumerate(lasts):
        dataset.append((np.array([float(i)]), np.zeros(1), 1., np.array([i + 1.]), False, bool(last)), {})


def fill_grid(dataset, active_steps, episode_ends):
    n_envs = len(active_steps)
    for t in range(max(active_steps)):
        mask = np.array([t < steps for steps in active_steps])
        last = np.array([(t, e) in episode_ends for e in range(n_envs)])
        step = (np.full((n_envs, 1), float(t)), np.zeros((n_envs, 1)), np.arange(n_envs) + 10. * t,
                np.full((n_envs, 1), t + 1.), np.zeros(n_envs, dtype=bool), last)
        dataset.append_vectorized(step, [{}] * n_envs, mask)


def make_block(lasts, start, continuing=False):
    n = len(lasts)
    states = np.arange(start, start + n, dtype=float)[:, None]
    return Dataset.from_array(states, np.zeros((n, 1)), np.ones(n), states + 1, np.zeros(n, dtype=bool),
                              np.array(lasts, dtype=bool), gamma=0.9, continuing=continuing)


def make_history_manager(history_length):
    mdp_info = MDPInfo(Box(np.full(1, -100.), np.full(1, 100.), (1,)), Box(-np.ones(1), np.ones(1), (1,)), 0.9, 100)
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')
    return HistoryManager.default_streams(mdp_info, agent_info, history_length=history_length)


def test_parse_forces_last_on_the_final_row_and_leaves_the_stored_column():
    dataset = make_dataset([False, False, True, False, False])

    assert np.array_equal(dataset.parse()[5], np.array([False, False, True, False, True]))
    assert np.array_equal(dataset.last, np.array([False, False, True, False, False]))


def test_parse_of_a_dataset_ending_on_an_episode_end_is_the_stored_last():
    dataset = make_dataset([False, True, False, True])

    assert np.array_equal(dataset.parse()[5], np.array([False, True, False, True]))


def test_to_backend_converts_the_stored_last():
    dataset = make_dataset([False, False, True, False, False]).to_backend('torch')

    assert torch.equal(dataset.last, torch.tensor([False, False, True, False, False]))
    assert torch.equal(dataset.parse()[5], torch.tensor([False, False, True, False, True]))


def test_contiguous_slice_starting_mid_episode_keeps_the_episode_structure():
    dataset = make_dataset([False, False, True, False, False])

    mid_episode = dataset[1:]

    assert np.array_equal(mid_episode.parse()[5], np.array([False, True, False, True]))


def test_add_stitches_consecutive_blocks_and_not_unrelated_ones():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, False])
    first_block = dataset.copy()
    dataset.clear()
    append_rows(dataset, [False, False])
    second_block = dataset.copy()

    stitched = first_block + second_block
    unrelated = first_block + first_block.copy()

    assert np.array_equal(stitched.parse()[5], np.array([False, False, False, False, True]))
    assert np.array_equal(unrelated.parse()[5], np.array([False, False, True, False, False, True]))
    assert np.array_equal(unrelated.last, np.zeros(6, dtype=bool))


def test_clear_after_an_episode_end_starts_a_fresh_stream():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, True])
    dataset.clear()
    append_rows(dataset, [False, False])

    assert np.array_equal(dataset.get_init_states(), np.array([[0.]]))
    assert dataset.n_episodes == 0
    assert len(dataset.episodes_length) == 0


def test_continuing_sequential_block_has_no_initial_state_and_counts_only_its_episode_end():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, False])
    dataset.clear()
    append_rows(dataset, [False, True])

    assert len(dataset.get_init_states()) == 0
    assert dataset.n_episodes == 1
    assert np.array_equal(dataset.episodes_length, np.array([2]))
    assert np.array_equal(dataset.compute_J(), np.array([2.]))


def test_flat_block_of_a_vectorized_dataset_segments_its_episodes():
    dataset = VectorizedDataset(make_grid_info(3), n_steps=12)
    fill_grid(dataset, active_steps=[4, 2, 3], episode_ends={(1, 0), (0, 2)})

    flat = dataset.flatten()

    assert np.array_equal(flat.parse()[5], np.array([False, True, False, True, False, True, True, False, True]))
    assert flat.n_episodes == 2
    assert np.array_equal(flat.compute_J(), np.array([10., 2.]))
    assert np.array_equal(flat.compute_J(skip_incomplete=False), np.array([10., 50., 12., 2., 34.]))
    assert np.array_equal(flat.episodes_length, np.array([2, 1]))
    assert np.array_equal(flat.get_init_states(), np.array([[0.], [2.], [0.], [0.], [1.]]))

    first_episode = flat.select_first_episodes(1)
    both_episodes = flat.select_first_episodes(2)

    assert np.array_equal(first_episode.reward, np.array([0., 10.]))
    assert np.array_equal(first_episode.parse()[5], np.array([False, True]))
    assert np.array_equal(both_episodes.reward, np.array([0., 10., 2.]))
    assert np.array_equal(both_episodes.parse()[5], np.array([False, True, True]))
    assert both_episodes.n_episodes == 2


def test_leftover_row_keeps_the_environment_memory():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[3, 3], episode_ends=set())

    dataset.flatten(5)
    n_carry = dataset.clear(keep_leftovers=True)
    fill_grid(dataset, active_steps=[1, 1], episode_ends=set())
    rest = dataset.flatten()

    assert n_carry == 1
    assert np.array_equal(rest.parse()[5], np.array([True, False, True]))


def test_list_backend_flat_block_segments_on_the_boundary():
    info = DatasetInfo(env_backend='list', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None, n_envs=2)
    dataset = VectorizedDataset(info, n_steps=12)
    fill_grid(dataset, active_steps=[2, 2], episode_ends={(1, 0)})

    flat = dataset.flatten()

    assert flat.array_backend.get_backend_name() == 'list'
    assert flat.parse()[5] == [False, True, False, True]
    assert flat.n_episodes == 1
    assert np.array_equal(flat.compute_J(), np.array([10.]))
    assert np.array_equal(flat.episodes_length, np.array([2]))


def test_continuing_block_stitches_to_an_open_tail():
    first = make_block([False, True, False], 0)
    second = make_block([False, True], 3, continuing=True)

    joined = first + second
    first += second

    for dataset in (joined, first):
        assert np.array_equal(dataset.parse()[5], np.array([False, True, False, False, True]))
        assert np.array_equal(dataset.episodes_length, np.array([2, 3]))
        assert np.array_equal(dataset.get_init_states(), np.array([[0.], [2.]]))


def test_standalone_block_cuts_the_open_tail():
    joined = make_block([False, True, False], 0) + make_block([False, True], 3)

    assert np.array_equal(joined.parse()[5], np.array([False, True, True, False, True]))
    assert np.array_equal(joined.last, np.array([False, True, False, False, True]))
    assert np.array_equal(joined.episodes_length, np.array([2, 2]))
    assert np.array_equal(joined.get_init_states(), np.array([[0.], [2.], [3.]]))
    assert joined.n_episodes == 2


def test_continuing_block_after_a_closed_tail_cannot_be_appended():
    closed = make_block([False, True], 0)

    with pytest.raises(ValueError):
        closed + make_block([False], 2, continuing=True)


def test_consume_rejects_fewer_steps_than_environments():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[2, 2], episode_ends=set())

    with pytest.raises(AssertionError):
        dataset.consume(1)
    with pytest.raises(AssertionError):
        dataset.flatten(1)

    assert np.array_equal(dataset.mask, np.ones((2, 2), dtype=bool))
    assert np.array_equal(dataset.flatten(2).reward, np.array([0., 1.]))


def test_empty_block_is_neutral_in_a_concatenation():
    first = make_block([False, True, False], 0)
    second = make_block([False, True], 3, continuing=True)
    vectorized = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(vectorized, active_steps=[2, 2], episode_ends=set())
    first_flat, second_flat = vectorized.flatten(2), vectorized.flatten()
    empty_flat = VectorizedDataset(make_grid_info(2), n_steps=12).flatten()

    with_empty_slice = (first + first[0:0]) + second
    with_empty_flat = first_flat + empty_flat + second_flat

    assert np.array_equal(with_empty_slice.parse()[5], np.array([False, True, False, False, True]))
    assert np.array_equal(with_empty_slice.parse()[5], (first + second).parse()[5])
    assert np.array_equal(with_empty_flat.parse()[5], np.array([True, True, True, True]))
    assert np.array_equal(with_empty_flat.parse()[5], (first_flat + second_flat).parse()[5])


def test_continuing_block_survives_a_backend_conversion():
    joined = (make_block([False, True, False], 0) + make_block([False, True], 3, continuing=True)).to_backend('torch')

    assert torch.equal(joined.parse()[5], torch.tensor([False, True, False, False, True]))
    assert torch.equal(joined.episodes_length, torch.tensor([2, 3]))


def test_slice_is_a_standalone_block():
    dataset = make_block([False, False, False, True, False, False], 0)
    history_manager = make_history_manager(3)

    mid_episode = dataset[2:]
    rejoined = dataset[:2] + dataset[2:]

    assert np.array_equal(history_manager.parse_state(mid_episode)[0], np.array([[0.], [0.], [2.]]))
    assert np.array_equal(history_manager.parse_state(dataset[4:])[0], np.array([[0.], [0.], [4.]]))
    assert np.array_equal(mid_episode.parse()[5], np.array([False, True, False, True]))
    assert np.array_equal(mid_episode.get_init_states(), np.array([[4.]]))
    assert len(mid_episode.history_state) == 0
    assert np.array_equal(rejoined.parse()[5], np.array([False, True, False, True, False, True]))
    with pytest.raises(ValueError):
        mid_episode + make_block([True], 6, continuing=True)


def test_scattered_view_ends_a_segment_at_every_row():
    dataset = make_block([False, True, False, False], 0)

    view = dataset[np.array([0, 1, 3])]

    assert np.array_equal(view.parse()[5], np.array([True, True, True]))
    assert np.array_equal(view.last, np.array([False, True, False]))
    assert np.array_equal(view.get_init_states(), np.array([[0.]]))


def test_consumed_grids_stacked_flatten_like_the_whole_grid():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[3, 3], episode_ends={(1, 0)})

    collected = dataset.consume(4).copy()
    collected.reserve(8)
    n_carry = dataset.clear(keep_leftovers=True)
    collected += dataset.consume()
    flat = collected.flatten()

    assert n_carry == 2
    assert np.array_equal(flat.reward, np.array([0., 10., 20., 1., 11., 21.]))
    assert np.array_equal(flat.parse()[5], np.array([False, True, True, False, False, True]))
    assert np.array_equal(flat.get_init_states(), np.array([[0.], [2.], [0.]]))


def test_flat_blocks_of_a_vectorized_dataset_keep_their_continuing_chunk():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[3, 3], episode_ends={(1, 0)})

    first = dataset.flatten(4)
    dataset.clear(keep_leftovers=True)
    second = dataset.flatten()
    joined = first + second

    assert np.array_equal(joined.parse()[5], np.array([False, True, False, True, True, True]))
    assert joined.n_episodes == 1
    assert np.array_equal(joined.get_init_states(), np.array([[0.], [0.], [2.]]))


def test_nstep_return_bootstraps_at_a_cut_segment_end_but_not_at_the_buffer_end():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[3, 3], episode_ends=set())
    history_manager = make_history_manager(1)

    flat = dataset.flatten()
    _, _, reward, _, _, _, extra = history_manager.parse_nstep_history(flat, gamma=0.5, n_steps_return=2)

    assert np.array_equal(extra['anchor'], np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(extra['endpoint'], np.array([1, 2, 2, 4, 5]))
    assert np.array_equal(reward, np.array([5., 20., 20., 6.5, 21.5]))
