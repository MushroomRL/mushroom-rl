import torch

from mushroom_rl.core._impl.torch_dataset import TorchDataset


def make_dataset(capacity=8):
    shapes = [(capacity, 2), (capacity,), (capacity,)]
    dtypes = [torch.float, torch.float, torch.bool]
    return TorchDataset(shapes, dtypes)


def test_torch_dataset_append_and_column():
    dataset = make_dataset()
    dataset.append(torch.tensor([0.0, 1.0]), 0.5, False)
    dataset.append(torch.tensor([2.0, 3.0]), 1.5, True)

    assert len(dataset) == 2
    assert dataset.n_columns == 3
    assert torch.equal(dataset.column(0), torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert torch.equal(dataset.column(1), torch.tensor([0.5, 1.5]))
    assert torch.equal(dataset.column(2), torch.tensor([False, True]))


def test_torch_dataset_clear():
    dataset = make_dataset()
    dataset.append(torch.tensor([0.0, 0.0]), 0.0, False)

    dataset.clear()

    assert len(dataset) == 0


def test_torch_dataset_get_view():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(torch.tensor([float(i), float(i)]), float(i), i == 3)

    view = dataset.get_view(slice(1, 3))
    assert len(view) == 2
    assert torch.equal(view.column(1), torch.tensor([1.0, 2.0]))


def test_torch_dataset_get_view_copy_isolation():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(torch.tensor([float(i), float(i)]), float(i), False)

    view = dataset.get_view(slice(0, 2), copy=True)
    view.column(1)[0] = 99.0

    assert dataset.column(1)[0] == 0.0


def test_torch_dataset_add():
    a = make_dataset()
    a.append(torch.tensor([0.0, 0.0]), 0.0, False)
    a.append(torch.tensor([1.0, 1.0]), 1.0, True)
    b = make_dataset()
    b.append(torch.tensor([2.0, 2.0]), 2.0, True)

    result = a + b

    assert len(result) == 3
    assert torch.equal(result.column(1), torch.tensor([0.0, 1.0, 2.0]))


def test_torch_dataset_append_batch():
    a = make_dataset()
    a.append(torch.tensor([0.0, 0.0]), 0.0, False)
    b = make_dataset()
    b.append(torch.tensor([1.0, 1.0]), 1.0, True)
    b.append(torch.tensor([2.0, 2.0]), 2.0, True)

    a.append_batch(b)

    assert len(a) == 3
    assert torch.equal(a.column(1), torch.tensor([0.0, 1.0, 2.0]))


def test_torch_dataset_capacity():
    dataset = make_dataset(capacity=5)
    dataset.append(torch.tensor([0.0, 0.0]), 0.0, False)

    assert dataset.capacity == 5


def test_torch_dataset_append_batch_past_capacity_raises():
    a = make_dataset(capacity=2)
    a.append(torch.tensor([0.0, 0.0]), 0.0, False)
    a.append(torch.tensor([1.0, 1.0]), 1.0, True)
    b = make_dataset(capacity=2)
    b.append(torch.tensor([2.0, 2.0]), 2.0, True)

    caught = False
    try:
        a.append_batch(b)
    except AssertionError:
        caught = True
    assert caught


def test_torch_dataset_reserve_grows_and_preserves():
    dataset = make_dataset(capacity=2)
    dataset.append(torch.tensor([0.0, 1.0]), 0.5, False)
    dataset.append(torch.tensor([2.0, 3.0]), 1.5, True)

    dataset.reserve(6)

    assert dataset.capacity == 6
    assert len(dataset) == 2
    assert torch.equal(dataset.column(0), torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert torch.equal(dataset.column(1), torch.tensor([0.5, 1.5]))
    assert torch.equal(dataset.column(2), torch.tensor([False, True]))

    dataset.append(torch.tensor([4.0, 5.0]), 2.5, True)
    assert len(dataset) == 3
    assert torch.equal(dataset.column(1), torch.tensor([0.5, 1.5, 2.5]))


def test_torch_dataset_reserve_noop_when_enough():
    dataset = make_dataset(capacity=8)
    dataset.append(torch.tensor([0.0, 0.0]), 0.0, False)

    dataset.reserve(4)

    assert dataset.capacity == 8


def test_torch_dataset_n_episodes():
    dataset = make_dataset()
    for i in range(4):
        dataset.append(torch.tensor([0.0, 0.0]), 0.0, i in (1, 3))

    assert dataset.n_episodes(2) == 2


def test_torch_dataset_from_array():
    states = torch.arange(6).reshape(3, 2).float()
    rewards = torch.arange(3).float()
    lasts = torch.tensor([False, False, True])

    dataset = TorchDataset.from_array([states, rewards, lasts])

    assert len(dataset) == 3
    assert torch.equal(dataset.column(0), states)
    assert dataset.n_episodes(2) == 1


def test_torch_dataset_truncates_to_n_envs():
    dataset = TorchDataset([(4, 2)], [torch.float], n_envs=2)
    dataset.append(torch.tensor([10.0, 20.0, 30.0]))

    assert torch.equal(dataset.column(0), torch.tensor([[10.0, 20.0]]))
