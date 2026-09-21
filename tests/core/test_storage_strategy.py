import numpy as np
import torch

from mushroom_rl.core import Dataset
from mushroom_rl.core.dataset import DatasetInfo, VectorizedDataset
from mushroom_rl.core._impl import ContiguousRows, PointerRows


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
    assert isinstance(dataset.storage_strategy, ContiguousRows)


def test_contiguous_slice_keeps_contiguous_rows_and_records_the_continuation():
    dataset = make_dataset([False, False, True, False, False])

    after_episode_end = dataset[3:]
    mid_episode = dataset[1:]

    assert isinstance(after_episode_end.storage_strategy, ContiguousRows)
    assert not after_episode_end.storage_strategy.continues
    assert after_episode_end.storage_strategy.first_id == 3
    assert mid_episode.storage_strategy.continues
    assert mid_episode.storage_strategy.first_id == 1
    assert np.array_equal(mid_episode.parse()[5], np.array([False, True, False, True]))


def test_random_view_promotes_to_pointer_rows():
    dataset = make_dataset([False, False, True, False, False])

    view = dataset[np.array([0, 1, 3, 4])]

    assert isinstance(view.storage_strategy, PointerRows)
    assert np.array_equal(view.storage_strategy.row_id, np.array([0, 1, 3, 4]))
    assert np.array_equal(view.storage_strategy.prev_id, np.array([-1, 0, -1, 3]))
    assert np.array_equal(view.parse()[5], np.array([False, True, False, True]))
    assert np.array_equal(view.last, np.array([False, False, False, False]))


def test_contiguous_and_pointer_rows_agree_on_contiguous_data():
    lasts = np.array([False, True, False, False, True, False])
    contiguous = ContiguousRows()
    pointers = contiguous.promote(lasts)

    assert np.array_equal(contiguous.segment_ends(lasts), pointers.segment_ends(lasts))
    assert np.array_equal(contiguous.segment_ends(lasts), np.array([False, True, False, False, True, True]))
    positions, starts = contiguous.segment_starts(lasts)
    pointer_positions, pointer_starts = pointers.segment_starts(lasts)
    assert np.array_equal(positions, np.array([0, 2, 5]))
    assert np.array_equal(positions, pointer_positions)
    assert np.array_equal(starts, np.array([True, True, True]))
    assert np.array_equal(starts, pointer_starts)


def test_add_stitches_consecutive_blocks_and_promotes_unrelated_ones():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, False])
    first_block = dataset.copy()
    dataset.clear()
    append_rows(dataset, [False, False])
    second_block = dataset.copy()

    assert second_block.storage_strategy.continues
    assert second_block.storage_strategy.first_id == 3

    stitched = first_block + second_block
    unrelated = first_block + first_block.copy()

    assert isinstance(stitched.storage_strategy, ContiguousRows)
    assert np.array_equal(stitched.parse()[5], np.array([False, False, False, False, True]))
    assert isinstance(unrelated.storage_strategy, PointerRows)
    assert np.array_equal(unrelated.parse()[5], np.array([False, False, True, False, False, True]))
    assert np.array_equal(unrelated.last, np.zeros(6, dtype=bool))


def test_clear_after_an_episode_end_starts_a_fresh_stream():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, True])
    dataset.clear()
    append_rows(dataset, [False, False])

    assert not dataset.storage_strategy.continues
    assert np.array_equal(dataset.get_init_states(), np.array([[0.]]))
    assert dataset.n_episodes == 0
    assert len(dataset.episodes_length) == 0


def test_continuing_sequential_block_has_no_initial_state_and_counts_only_its_episode_end():
    dataset = Dataset(make_sequential_info(), n_steps=10)
    append_rows(dataset, [False, False, False])
    dataset.clear()
    append_rows(dataset, [False, True])

    assert dataset.storage_strategy.continues
    assert len(dataset.get_init_states()) == 0
    assert dataset.n_episodes == 1
    assert np.array_equal(dataset.episodes_length, np.array([2]))
    assert np.array_equal(dataset.compute_J(), np.array([2.]))


def test_flat_block_of_a_vectorized_dataset_carries_pointers_and_segments():
    dataset = VectorizedDataset(make_grid_info(3), n_steps=12)
    fill_grid(dataset, active_steps=[4, 2, 3], episode_ends={(1, 0), (0, 2)})

    flat = dataset.flatten()

    assert isinstance(flat.storage_strategy, PointerRows)
    assert np.array_equal(flat.storage_strategy.row_id, np.arange(9))
    assert np.array_equal(flat.storage_strategy.prev_id, np.array([-1, 0, -1, 2, -1, 4, -1, -1, 7]))
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


def test_second_flatten_continues_the_stream_of_every_environment():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[2, 2], episode_ends=set())
    first = dataset.flatten()
    dataset.clear()
    fill_grid(dataset, active_steps=[1, 1], episode_ends=set())

    second = dataset.flatten()
    positions, is_episode_start = second.storage_strategy.segment_starts(second.last)

    assert np.array_equal(first.storage_strategy.prev_id, np.array([-1, 0, -1, 2]))
    assert np.array_equal(second.storage_strategy.row_id, np.array([4, 5]))
    assert np.array_equal(second.storage_strategy.prev_id, np.array([1, 3]))
    assert np.array_equal(second.parse()[5], np.array([True, True]))
    assert np.array_equal(positions, np.array([0, 1]))
    assert np.array_equal(is_episode_start, np.array([False, False]))


def test_leftover_row_keeps_the_environment_memory():
    dataset = VectorizedDataset(make_grid_info(2), n_steps=12)
    fill_grid(dataset, active_steps=[3, 3], episode_ends=set())

    consumed = dataset.flatten(5)
    n_carry = dataset.clear(keep_leftovers=True)
    fill_grid(dataset, active_steps=[1, 1], episode_ends=set())
    rest = dataset.flatten()

    assert n_carry == 1
    assert np.array_equal(consumed.storage_strategy.prev_id, np.array([-1, 0, 1, -1, 3]))
    assert np.array_equal(rest.storage_strategy.row_id, np.array([5, 6, 7]))
    assert np.array_equal(rest.storage_strategy.prev_id, np.array([2, 4, 6]))
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
