import numpy as np

from mushroom_rl.core._impl.list_dataset import ListDataset


def build_arrays(n, terminal_last=True):
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float)
    next_states = states + 1
    absorbing = np.zeros(n)
    last = np.zeros(n)
    if terminal_last:
        last[-1] = 1
    return states, actions, rewards, next_states, absorbing, last


def test_list_dataset_non_stateful():
    n = 5
    states, actions, rewards, next_states, absorbing, last = build_arrays(n)

    dataset = ListDataset.from_array(states, actions, rewards, next_states, absorbing, last)

    assert len(dataset) == n
    assert not dataset.is_stateful
    assert dataset.mask is None

    assert np.array_equal(dataset.state, states)
    assert np.array_equal(dataset.action, actions)
    assert dataset.reward == list(rewards)
    assert np.array_equal(dataset.next_state, next_states)
    assert dataset.absorbing == [False] * n
    assert dataset.last == [False, False, False, False, True]

    assert np.array_equal(dataset[0][0], states[0])
    assert dataset.n_episodes == 1


def test_list_dataset_n_episodes_open():
    n = 4
    arrays = build_arrays(n, terminal_last=False)
    dataset = ListDataset.from_array(*arrays)

    assert dataset.n_episodes == 1


def test_list_dataset_stateful():
    n = 4
    states, actions, rewards, next_states, absorbing, last = build_arrays(n)
    policy_states = np.arange(n).astype(float)
    policy_next_states = np.arange(n).astype(float) + 1

    dataset = ListDataset.from_array(states, actions, rewards, next_states, absorbing, last,
                                     policy_states, policy_next_states)

    assert dataset.is_stateful
    assert dataset.policy_state == list(policy_states)
    assert dataset.policy_next_state == list(policy_next_states)


def test_list_dataset_append_and_clear():
    n = 3
    arrays = build_arrays(n)
    dataset = ListDataset.from_array(*arrays)

    dataset.append(np.zeros(2), np.zeros(1), 1.0, np.ones(2), False, False)
    assert len(dataset) == n + 1

    other = ListDataset.from_array(*build_arrays(2))
    dataset.append_batch(other)
    assert len(dataset) == n + 1 + 2

    dataset.clear()
    assert len(dataset) == 0


def test_list_dataset_add():
    n_a = 3
    n_b = 2
    arrays_a = build_arrays(n_a, terminal_last=False)
    arrays_b = build_arrays(n_b)
    dataset_a = ListDataset.from_array(*arrays_a)
    dataset_b = ListDataset.from_array(*arrays_b)

    result = dataset_a + dataset_b

    assert len(result) == n_a + n_b
    assert result.last == [False, False, True, False, True]
    assert np.array_equal(result.state[:n_a], dataset_a.state)
    assert np.array_equal(result.state[n_a:], dataset_b.state)
    assert dataset_a.last == [False, False, False]


def test_list_dataset_get_view():
    n = 6
    arrays = build_arrays(n)
    dataset = ListDataset.from_array(*arrays)

    view_slice = dataset.get_view(slice(0, 3))
    assert len(view_slice) == 3

    view_index = dataset.get_view([0, 2, 4])
    assert len(view_index) == 3

    view_copy = dataset.get_view(slice(0, 2), copy=True)
    assert len(view_copy) == 2


def test_list_dataset_get_view_mask():
    dataset = ListDataset(is_stateful=False, is_vectorized=True)
    for i in range(4):
        dataset.append(np.array([float(i)]), np.zeros(1), 0.0, np.array([float(i)]), False, False,
                       mask=np.array([True, i % 2 == 0]))

    view_slice = dataset.get_view(slice(0, 2))
    assert len(view_slice.mask) == 2
    assert np.array_equal(view_slice.mask[0], np.array([True, True]))

    view_index = dataset.get_view([0, 3])
    assert len(view_index.mask) == 2
    assert np.array_equal(view_index.mask[1], np.array([True, False]))


def test_list_dataset_mask_setter():
    n = 3
    arrays = build_arrays(n)
    dataset = ListDataset.from_array(*arrays)

    mask = np.ones(n, dtype=bool)
    dataset.mask = mask
    assert np.array_equal(dataset.mask, mask)
