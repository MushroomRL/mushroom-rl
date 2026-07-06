import numpy as np

from mushroom_rl.core._impl.numpy_dataset import NumpyDataset


def make_dataset(capacity=8):
    shapes = [(capacity, 2), (capacity,), (capacity,)]
    dtypes = [float, float, bool]
    return NumpyDataset(shapes, dtypes)


def test_numpy_dataset_append_and_column():
    dataset = make_dataset()
    dataset.append(np.array([0.0, 1.0]), 0.5, False)
    dataset.append(np.array([2.0, 3.0]), 1.5, True)

    assert len(dataset) == 2
    assert dataset.n_columns == 3
    assert np.array_equal(dataset.column(0), np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert np.array_equal(dataset.column(1), np.array([0.5, 1.5]))
    assert np.array_equal(dataset.column(2), np.array([False, True]))
    assert len(dataset.columns) == 3


def test_numpy_dataset_getitem_step():
    dataset = make_dataset()
    dataset.append(np.array([4.0, 5.0]), 2.0, True)

    step = dataset[0]

    assert np.array_equal(step[0], np.array([4.0, 5.0]))
    assert step[1] == 2.0
    assert bool(step[2])


def test_numpy_dataset_clear():
    dataset = make_dataset()
    dataset.append(np.array([0.0, 0.0]), 0.0, False)

    dataset.clear()

    assert len(dataset) == 0


def test_numpy_dataset_get_view():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(np.array([float(i), float(i)]), float(i), i == 3)

    view = dataset.get_view(slice(1, 3))
    assert len(view) == 2
    assert np.array_equal(view.column(1), np.array([1.0, 2.0]))

    view_idx = dataset.get_view(np.array([0, 3]))
    assert np.array_equal(view_idx.column(1), np.array([0.0, 3.0]))


def test_numpy_dataset_get_view_copy_isolation():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(np.array([float(i), float(i)]), float(i), False)

    view = dataset.get_view(slice(0, 2), copy=True)
    view.column(1)[0] = 99.0

    assert dataset.column(1)[0] == 0.0


def test_numpy_dataset_add():
    a = make_dataset()
    a.append(np.array([0.0, 0.0]), 0.0, False)
    a.append(np.array([1.0, 1.0]), 1.0, True)
    b = make_dataset()
    b.append(np.array([2.0, 2.0]), 2.0, True)

    result = a + b

    assert len(result) == 3
    assert np.array_equal(result.column(1), np.array([0.0, 1.0, 2.0]))


def test_numpy_dataset_append_batch():
    a = make_dataset()
    a.append(np.array([0.0, 0.0]), 0.0, False)
    b = make_dataset()
    b.append(np.array([1.0, 1.0]), 1.0, True)
    b.append(np.array([2.0, 2.0]), 2.0, True)

    a.append_batch(b)

    assert len(a) == 3
    assert np.array_equal(a.column(1), np.array([0.0, 1.0, 2.0]))


def test_numpy_dataset_n_episodes():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(np.array([0.0, 0.0]), 0.0, i in (1, 3))
    assert dataset.n_episodes(2) == 2

    open_dataset = make_dataset()
    for i in range(3):
        open_dataset.append(np.array([0.0, 0.0]), 0.0, False)
    assert open_dataset.n_episodes(2) == 1


def test_numpy_dataset_from_array():
    states = np.arange(6).reshape(3, 2).astype(float)
    rewards = np.arange(3).astype(float)
    lasts = np.array([False, False, True])

    dataset = NumpyDataset.from_array([states, rewards, lasts])

    assert len(dataset) == 3
    assert np.array_equal(dataset.column(0), states)
    assert dataset.n_episodes(2) == 1


def test_numpy_dataset_truncates_to_n_envs():
    dataset = NumpyDataset([(4, 2)], [float], n_envs=2)
    dataset.append(np.array([10.0, 20.0, 30.0]))
    dataset.append(np.array([1.0, 2.0, 3.0]))

    assert np.array_equal(dataset.column(0), np.array([[10.0, 20.0], [1.0, 2.0]]))
