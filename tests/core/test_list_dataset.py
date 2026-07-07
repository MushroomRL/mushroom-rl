import numpy as np

from mushroom_rl.core._impl.list_dataset import ListDataset


def build_columns(n, terminal_last=True):
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float)
    next_states = states + 1
    absorbing = np.zeros(n)
    last = np.zeros(n)
    if terminal_last:
        last[-1] = 1
    return [states, actions, rewards, next_states, absorbing, last]


def test_list_dataset_from_array_columns():
    n = 5
    columns = build_columns(n)
    dataset = ListDataset.from_array(columns)

    assert len(dataset) == n
    assert dataset.n_columns == 6
    assert np.array_equal(np.array(dataset.column(0)), columns[0])
    assert dataset.column(2) == list(columns[2])
    assert dataset.column(5)[-1] == 1.0
    assert len(dataset.data) == 6


def test_list_dataset_append_and_len():
    dataset = ListDataset(6)

    dataset.append(np.zeros(2), np.zeros(1), 1.0, np.ones(2), False, False)
    dataset.append(np.ones(2), np.ones(1), 2.0, np.ones(2) * 2, False, True)

    assert len(dataset) == 2
    assert dataset.column(2) == [1.0, 2.0]
    assert dataset.column(5) == [False, True]


def test_list_dataset_append_isolation():
    dataset = ListDataset(6)

    state = np.zeros(2)
    dataset.append(state, np.zeros(1), 0.0, np.zeros(2), False, False)
    state[0] = 99.0

    assert dataset.column(0)[0][0] == 0.0


def test_list_dataset_append_batch():
    dataset = ListDataset.from_array(build_columns(3))
    other = ListDataset.from_array(build_columns(2))

    dataset.append_batch(other)

    assert len(dataset) == 5
    assert dataset.column(2) == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_list_dataset_clear():
    dataset = ListDataset.from_array(build_columns(3))

    dataset.clear()

    assert len(dataset) == 0
    assert dataset.n_columns == 6


def test_list_dataset_get_view():
    dataset = ListDataset.from_array(build_columns(6))

    view_slice = dataset.get_view(slice(0, 3))
    assert len(view_slice) == 3
    assert view_slice.column(2) == [0.0, 1.0, 2.0]

    view_index = dataset.get_view([0, 2, 4])
    assert len(view_index) == 3
    assert view_index.column(2) == [0.0, 2.0, 4.0]


def test_list_dataset_get_view_copy_isolation():
    dataset = ListDataset.from_array(build_columns(4))

    view_copy = dataset.get_view(slice(0, 2), copy=True)
    view_copy.column(0)[0][0] = 123.0

    assert dataset.column(0)[0][0] == 0.0


def test_list_dataset_getitem_returns_step():
    dataset = ListDataset.from_array(build_columns(4))

    step = dataset[1]

    assert len(step) == 6
    assert np.array_equal(step[0], np.array([2.0, 3.0]))
    assert step[2] == 1.0


def test_list_dataset_add():
    dataset_a = ListDataset.from_array(build_columns(3, terminal_last=False))
    dataset_b = ListDataset.from_array(build_columns(2))

    result = dataset_a + dataset_b

    assert len(result) == 5
    assert result.column(2) == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert dataset_a.column(5) == [0.0, 0.0, 0.0]


def test_list_dataset_column_single():
    dataset = ListDataset(1)
    dataset.append(np.array([True, False]))
    dataset.append(np.array([True, True]))

    assert np.array_equal(np.array(dataset.column()), np.array([[True, False], [True, True]]))


def test_list_dataset_n_episodes_terminal():
    dataset = ListDataset.from_array(build_columns(5))

    assert dataset.n_episodes(5) == 1


def test_list_dataset_n_episodes_open():
    dataset = ListDataset.from_array(build_columns(4, terminal_last=False))

    assert dataset.n_episodes(5) == 1


def test_list_dataset_ragged_content():
    dataset = ListDataset(6)
    dataset.append([0.0], 0, 0.0, [0.0, 1.0], False, False)
    dataset.append([0.0, 1.0, 2.0], 1, 1.0, [0.0], False, True)

    assert dataset.column(0)[0] == [0.0]
    assert dataset.column(0)[1] == [0.0, 1.0, 2.0]
    assert dataset.n_episodes(5) == 1


def test_list_dataset_truncates_to_n_envs():
    dataset = ListDataset(1, n_envs=2)
    dataset.append([10.0, 20.0, 30.0])

    assert dataset.column(0)[0] == [10.0, 20.0]
