import numpy as np
import pytest
import torch

from mushroom_rl.core import MDPInfo, AgentInfo, Dataset
from mushroom_rl.core.circular_dataset import CircularDataset
from mushroom_rl.core.dataset_info import DatasetInfo
from mushroom_rl.core.vectorized_dataset import VectorizedDataset
from mushroom_rl.core.spaces import Box


def make_block(states, lasts, rewards=None, continuing=False):
    n = len(states)
    states = np.array(states, dtype=float)[:, None]
    rewards = np.zeros(n) if rewards is None else np.array(rewards, dtype=float)
    return Dataset.from_array(states, np.zeros((n, 1)), rewards, states + 0.5, np.zeros(n, dtype=bool),
                              np.array(lasts, dtype=bool), continuing=continuing)


def make_wrapped_buffer():
    # episode A = 10, 11, 12 (reward 1 each), then episode B = 20, 21, 22, 23 (reward 2 each): the buffer holds
    # [22, 23, 12, 20, 21], A's first two steps overwritten
    first = make_block([10, 11, 12], [False, False, True], rewards=[1, 1, 1])
    second = make_block([20, 21, 22, 23], [False, False, False, True], rewards=[2, 2, 2, 2])
    buffer = CircularDataset(first._dataset_info, 5)
    buffer.append_replay_batch(first)
    buffer.append_replay_batch(second)
    return buffer


def make_step(value, last):
    return np.array([float(value)]), np.zeros(1), float(value), np.array([value + 0.5]), False, last


def make_linked_buffer():
    # two environments stepped four times, written in two blocks of two steps each: env 0 stores 0..3 with reward
    # t, env 1 stores 10..13 with reward 100 + t; the buffer holds [12, 13, 10, 11, 2, 3], env 0's first block
    # overwritten, both episodes open
    info = DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=100, gamma=0.5, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None, n_envs=2)
    grid = VectorizedDataset(info, n_steps=40)
    blocks = list()
    for steps in ([0, 1], [2, 3]):
        for t in steps:
            values = 10. * np.arange(2) + t
            step = (values[:, None], np.zeros((2, 1)), 100. * np.arange(2) + t, values[:, None] + 0.5,
                    np.zeros(2, dtype=bool), np.zeros(2, dtype=bool))
            grid.append_vectorized(step, [{}] * 2, np.ones(2, dtype=bool))
        blocks.append(grid.flatten())
        grid.clear(keep_leftovers=True)
    buffer = CircularDataset(blocks[0]._dataset_info, 6)
    for block in blocks:
        buffer.append_replay_batch(block)
    return buffer


def test_append_batch_writes_at_the_write_head_and_wraps():
    first = make_block([10, 11, 12], [False, False, True])
    second = make_block([20, 21, 22, 23], [False, False, False, True])
    buffer = CircularDataset(first._dataset_info, 5)

    buffer.append_batch(first)
    buffer.append_batch(second)

    assert np.array_equal(buffer.state[:, 0], np.array([22., 23., 12., 20., 21.]))
    assert np.array_equal(buffer.last, np.array([False, True, True, False, False]))
    assert buffer.write_head == 2 and buffer.full and len(buffer) == 5


def test_append_batch_of_an_empty_dataset_keeps_the_open_episode():
    buffer = CircularDataset(make_block([0], [False])._dataset_info, 4)
    buffer.append_batch(make_block([0, 1], [False, False]))

    buffer.append_batch(make_block([], []))
    buffer.append_batch(make_block([2], [True], continuing=True))

    assert np.array_equal(buffer.state[:, 0], np.array([0., 1., 2.]))
    assert np.array_equal(buffer.last, np.array([False, False, True]))
    assert np.array_equal(buffer.episodes_length, np.array([3]))


def test_iadd_writes_in_place_and_add_writes_in_a_copy():
    buffer = make_wrapped_buffer()
    block = make_block([30, 31], [False, True])

    result = buffer + block

    assert type(result) is CircularDataset and result.max_size == 5
    assert np.array_equal(result.state[:, 0], np.array([22., 23., 30., 31., 21.]))
    assert np.array_equal(buffer.state[:, 0], np.array([22., 23., 12., 20., 21.]))

    same = buffer
    buffer += block

    assert buffer is same
    assert np.array_equal(buffer.state[:, 0], np.array([22., 23., 30., 31., 21.]))
    assert buffer.write_head == 4


def test_append_continues_the_open_episode_and_wraps():
    buffer = CircularDataset(make_block([0], [False])._dataset_info, 3)

    for value, last in [(1, False), (2, False), (3, True), (4, False), (5, False)]:
        buffer.append(make_step(value, last), {})

    assert np.array_equal(buffer.state[:, 0], np.array([4., 5., 3.]))
    assert np.array_equal(buffer.last, np.array([False, False, True]))
    assert buffer.write_head == 2 and buffer.full
    assert buffer.links is None
    assert np.array_equal(buffer.compute_J(skip_incomplete=False), np.array([3., 9.]))
    assert np.array_equal(buffer.get_init_states()[:, 0], np.array([4.]))

    buffer.append_batch(make_block([6], [True], rewards=[6], continuing=True))

    assert np.array_equal(buffer.state[:, 0], np.array([4., 5., 6.]))
    assert np.array_equal(buffer.compute_J(), np.array([15.]))


def test_append_stores_the_policy_state():
    states = np.array([[1.], [2.]])
    buffer = CircularDataset.from_array(states, np.zeros((2, 1)), np.zeros(2), states, np.zeros(2, dtype=bool),
                                        np.array([False, False]), policy_state=np.array([[10.], [20.]]),
                                        policy_next_state=np.array([[20.], [30.]]))

    buffer.append(make_step(3, True) + (np.array([30.]), np.array([40.])), {})

    assert np.array_equal(buffer.state[:, 0], np.array([3., 2.]))
    assert np.array_equal(buffer.policy_state[:, 0], np.array([30., 20.]))
    assert np.array_equal(buffer.policy_next_state[:, 0], np.array([40., 30.]))


def test_append_replay_batch_wraps_the_policy_state():
    states = np.array([[1.], [2.]])
    buffer = CircularDataset.from_array(states, np.zeros((2, 1)), np.zeros(2), states, np.zeros(2, dtype=bool),
                                        np.array([False, True]), policy_state=np.array([[10.], [20.]]),
                                        policy_next_state=np.array([[11.], [21.]]), max_size=3)
    states = np.array([[3.], [4.]])
    block = Dataset.from_array(states, np.zeros((2, 1)), np.zeros(2), states, np.zeros(2, dtype=bool),
                               np.array([False, True]), policy_state=np.array([[30.], [40.]]),
                               policy_next_state=np.array([[31.], [41.]]))

    buffer.append_replay_batch(block)

    assert np.array_equal(buffer.state[:, 0], np.array([4., 2., 3.]))
    assert np.array_equal(buffer.policy_state[:, 0], np.array([40., 20., 30.]))
    assert np.array_equal(buffer.policy_next_state[:, 0], np.array([41., 21., 31.]))


def test_append_after_a_write_leaving_several_episodes_open_raises():
    buffer = make_linked_buffer()

    with pytest.raises(ValueError):
        buffer.append(make_step(4, False), {})


def test_step_info_is_dropped():
    buffer = CircularDataset(make_block([0], [False])._dataset_info, 3)

    buffer.append(make_step(1, False), {'value': 1.})

    assert buffer.info == {}
    assert buffer.episode_info == {}
    assert buffer.theta_list == []


def test_append_episode_info_and_append_theta_raise():
    buffer = CircularDataset(make_block([0], [False])._dataset_info, 3)

    with pytest.raises(TypeError):
        buffer.append_episode_info({'seed': 1})
    with pytest.raises(TypeError):
        buffer.append_theta(np.ones(2))


def test_reserve_raises():
    buffer = make_wrapped_buffer()

    with pytest.raises(NotImplementedError):
        buffer.reserve(10)


def test_clear_moves_the_write_head_back_to_the_start():
    buffer = make_wrapped_buffer()

    buffer.clear()

    assert len(buffer) == 0 and buffer.write_head == 0 and not buffer.full

    buffer.append_replay_batch(make_block([10, 11, 12], [False, False, True]))

    assert np.array_equal(buffer.state[:, 0], np.array([10., 11., 12.]))
    assert buffer.write_head == 3
    assert np.array_equal(buffer.episodes_length, np.array([3]))
    assert np.array_equal(buffer.get_init_states()[:, 0], np.array([10.]))


def test_clear_drops_the_links_of_a_linked_ring():
    buffer = make_linked_buffer()

    buffer.clear()

    assert len(buffer) == 0 and buffer.write_head == 0 and not buffer.full
    assert np.array_equal(buffer.links[0], np.zeros(6, dtype=int))
    assert np.array_equal(buffer.links[1], np.zeros(6, dtype=int))


def test_views_are_plain_datasets_with_the_episode_structure_of_the_ring():
    buffer = make_wrapped_buffer()

    across_the_wrap = buffer[0:3]
    across_the_head = buffer[1:4]

    assert type(across_the_wrap) is Dataset and type(buffer[np.array([0, 3])]) is Dataset
    assert np.array_equal(across_the_wrap.state[:, 0], np.array([22., 23., 12.]))
    assert np.array_equal(across_the_wrap.episodes_length, np.array([2, 1]))
    assert len(across_the_wrap.get_init_states()) == 0
    assert np.array_equal(across_the_head.state[:, 0], np.array([23., 12., 20.]))
    assert np.array_equal(across_the_head.episodes_length, np.array([1, 1]))
    assert np.array_equal(across_the_head.get_init_states()[:, 0], np.array([20.]))


def test_per_episode_methods_follow_the_time_order():
    buffer = make_wrapped_buffer()

    assert np.array_equal(buffer.episodes_length, np.array([1, 4]))
    assert np.array_equal(buffer.undiscounted_return, np.array([1., 8.]))
    assert np.array_equal(buffer.compute_J(0.5), np.array([1., 3.75]))
    assert np.array_equal(buffer.get_init_states()[:, 0], np.array([20.]))
    assert buffer.compute_metrics()['n_episodes'] == 2 and buffer.compute_metrics()['mean_J'] == 4.5

    oldest = buffer.select_first_episodes(1)
    both = buffer.select_first_episodes(2)

    assert type(oldest) is Dataset
    assert np.array_equal(oldest.state[:, 0], np.array([12.]))
    assert np.array_equal(both.state[:, 0], np.array([12., 20., 21., 22., 23.]))
    with pytest.raises(IndexError):
        buffer.select_first_episodes(3)


def test_per_episode_methods_follow_the_episodes_of_a_linked_ring():
    buffer = make_linked_buffer()

    assert np.array_equal(buffer.state[:, 0], np.array([12., 13., 10., 11., 2., 3.]))
    assert np.array_equal(buffer.compute_J(skip_incomplete=False), np.array([406., 5.]))
    assert len(buffer.compute_J()) == 0 and len(buffer.episodes_length) == 0
    assert np.array_equal(buffer.get_init_states()[:, 0], np.array([10.]))
    assert np.array_equal(buffer[0:6].compute_J(skip_incomplete=False), np.array([205., 201., 5.]))


def test_to_backend_keeps_the_ring():
    buffer = make_wrapped_buffer()

    converted = buffer.to_backend('torch')

    assert buffer.to_backend('numpy') is buffer
    assert type(converted) is CircularDataset and converted.max_size == 5
    assert converted.write_head == 2 and converted.full
    assert torch.equal(converted.state[:, 0], torch.tensor([22., 23., 12., 20., 21.], dtype=converted.state.dtype))
    assert torch.equal(converted.episodes_length, torch.tensor([1, 4]))

    converted.append_replay_batch(make_block([30, 31], [False, True]))

    assert torch.equal(converted.state[:, 0], torch.tensor([22., 23., 30., 31., 21.], dtype=converted.state.dtype))


def test_to_backend_keeps_the_links_of_a_linked_ring():
    buffer = make_linked_buffer()

    converted = buffer.to_backend('torch')

    assert torch.equal(converted.links[0], torch.tensor([3, 1, 0, 1, 6, 1]))
    assert torch.equal(converted.compute_J(skip_incomplete=False),
                       torch.tensor([406., 5.], dtype=converted.reward.dtype))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_save_and_load_keep_the_links_of_a_linked_ring(tmpdir, backend):
    buffer = make_linked_buffer().to_backend(backend)
    path = tmpdir / 'linked_ring.msh'

    buffer.save(path)
    loaded = CircularDataset.load(path)

    assert loaded.write_head == buffer.write_head and loaded.full == buffer.full
    for original, restored in zip(buffer.links, loaded.links):
        assert type(restored) is type(original) and bool((restored == original).all())
    assert bool((loaded.compute_J(skip_incomplete=False) == buffer.compute_J(skip_incomplete=False)).all())

    for dataset in (buffer, loaded):
        dataset.append_replay_batch(make_block([30, 31], [False, True]))
    for original, restored in zip(buffer.links, loaded.links):
        assert bool((restored == original).all())


def test_from_array_writes_the_transitions_from_the_start_of_the_buffer():
    states = np.arange(3.)[:, None]
    buffer = CircularDataset.from_array(states, np.zeros((3, 1)), np.ones(3), states, np.zeros(3, dtype=bool),
                                        np.array([False, True, False]), max_size=5)
    continuing = CircularDataset.from_array(states[:2], np.zeros((2, 1)), np.ones(2), states[:2],
                                            np.zeros(2, dtype=bool), np.array([False, True]), continuing=True,
                                            max_size=3)

    assert type(buffer) is CircularDataset and buffer.max_size == 5 and buffer.write_head == 3
    assert np.array_equal(buffer.episodes_length, np.array([2]))
    assert np.array_equal(buffer.get_init_states()[:, 0], np.array([0., 2.]))
    assert continuing.max_size == 3 and not continuing.full
    assert np.array_equal(continuing.episodes_length, np.array([2]))
    assert len(continuing.get_init_states()) == 0
    with pytest.raises(ValueError):
        CircularDataset.from_array(states, np.zeros((3, 1)), np.ones(3), states, np.zeros(3, dtype=bool),
                                   np.array([False, True, False]), max_size=2)


def test_generate_builds_an_empty_buffer():
    mdp_info = MDPInfo(Box(-np.ones(1), np.ones(1), (1,)), Box(-np.ones(1), np.ones(1), (1,)), 0.9, 10)
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')

    buffer = CircularDataset.generate(mdp_info, agent_info, 4)
    buffer.append(make_step(1, True), {})

    assert type(buffer) is CircularDataset and buffer.max_size == 4
    assert len(buffer) == 1 and buffer.write_head == 1 and not buffer.full
    assert np.array_equal(buffer.state[:, 0], np.array([1.]))
