import numpy.random
import torch
import numpy as np
import pytest
from mushroom_rl.core.array_backend import ArrayBackend, NumpyBackend, TorchBackend, ListBackend


def test_get_array_backend():
    assert ArrayBackend.get_array_backend('numpy') is NumpyBackend
    assert ArrayBackend.get_array_backend('torch') is TorchBackend
    assert ArrayBackend.get_array_backend('list') is ListBackend

    with pytest.raises(AssertionError):
        ArrayBackend.get_array_backend(1)

    with pytest.raises(ValueError):
        ArrayBackend.get_array_backend('unknown')


def test_get_array_backend_from():
    assert ArrayBackend.get_array_backend_from(np.zeros(2)) is NumpyBackend
    assert ArrayBackend.get_array_backend_from(torch.zeros(2)) is TorchBackend
    assert ArrayBackend.get_array_backend_from([1, 2]) is ListBackend

    with pytest.raises(ValueError):
        ArrayBackend.get_array_backend_from(1)


def test_abstract_backend_not_implemented():
    calls = [
        lambda: ArrayBackend.get_backend_name(),
        lambda: ArrayBackend.get_backend_serialization(),
        lambda: ArrayBackend.convert_to_backend(ArrayBackend, None),
        lambda: ArrayBackend.to_numpy(None),
        lambda: ArrayBackend.to_torch(None),
        lambda: ArrayBackend.zeros(2, dtype=float),
        lambda: ArrayBackend.ones(2, dtype=float),
        lambda: ArrayBackend.zeros_like(None, dtype=float),
        lambda: ArrayBackend.ones_like(None, dtype=float),
        lambda: ArrayBackend.concatenate([], 0),
        lambda: ArrayBackend.where(None),
        lambda: ArrayBackend.squeeze(None, 0),
        lambda: ArrayBackend.expand_dims(None, 0),
        lambda: ArrayBackend.size(None),
        lambda: ArrayBackend.rand(2),
        lambda: ArrayBackend.randint(0, 1, 1),
        lambda: ArrayBackend.multinomial(None),
        lambda: ArrayBackend.uniform(0, 1),
        lambda: ArrayBackend.arange(0, 1),
        lambda: ArrayBackend.abs(None),
        lambda: ArrayBackend.exp(None),
        lambda: ArrayBackend.clip(None, 0, 1),
        lambda: ArrayBackend.atleast_2d(None),
        lambda: ArrayBackend.copy(None),
        lambda: ArrayBackend.median(None),
        lambda: ArrayBackend.sqrt(None),
        lambda: ArrayBackend.from_list(None),
        lambda: ArrayBackend.pack_padded_sequence(None, None),
        lambda: ArrayBackend.flatten(None),
        lambda: ArrayBackend.empty(2),
        lambda: ArrayBackend.none(),
        lambda: ArrayBackend.shape(None),
        lambda: ArrayBackend.full(2, 0),
        lambda: ArrayBackend.nonzero(None),
        lambda: ArrayBackend.repeat(None, 1),
        lambda: ArrayBackend.inf(),
        lambda: ArrayBackend.maximum(None, None),
        lambda: ArrayBackend.minimum(None, None),
        lambda: ArrayBackend.max(None),
        lambda: ArrayBackend.min(None),
        lambda: ArrayBackend.norm(None),
        lambda: ArrayBackend.logical_and(None, None),
        lambda: ArrayBackend.sum(None),
        lambda: ArrayBackend.stack([], 0),
        lambda: ArrayBackend.to_backend_dtype(float),
    ]

    for call in calls:
        with pytest.raises(NotImplementedError):
            call()


def test_list_backend():
    assert ListBackend.get_backend_name() == 'list'
    assert ListBackend.get_backend_serialization() == 'numpy'

    assert np.array_equal(ListBackend.to_numpy([1, 2, 3]), np.array([1, 2, 3]))
    assert torch.equal(ListBackend.to_torch([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert ListBackend.to_torch(None) is None
    assert ListBackend.to_backend_dtype(np.float32) == np.float32
    raw_object = [1, 2]
    assert ListBackend.convert_to_backend(ListBackend, raw_object) is raw_object
    assert np.array_equal(ListBackend.as_array([1, 2, 3]), np.array([1, 2, 3]))
    kept = np.array([1, 2, 3])
    assert NumpyBackend.as_array(kept) is kept

    assert np.array_equal(ListBackend.zeros(2, 3, dtype=float), np.zeros((2, 3)))
    assert np.array_equal(ListBackend.ones(2, 3, dtype=float), np.ones((2, 3)))
    assert np.array_equal(ListBackend.zeros_like(np.ones(3)), np.zeros(3))
    assert np.array_equal(ListBackend.ones_like(np.zeros(3)), np.ones(3))

    with pytest.raises(ValueError):
        ListBackend.zeros(2, dtype=float, device='cuda')

    array = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(ListBackend.copy(array), array)

    nested = [[1, 2], [3, 4]]
    nested_copy = ListBackend.copy(nested)
    nested_copy[0].append(99)
    assert nested == [[1, 2], [3, 4]]

    assert ListBackend.median(array) == 2.0
    assert ListBackend.from_list(array) is array
    assert ListBackend.empty((3,)) == [None, None, None]
    assert ListBackend.empty((2, 2)) == [[None, None], [None, None]]
    assert ListBackend.none() is None
    assert ListBackend.shape([[1, 2], [3, 4]]) == (2, 2)
    assert ListBackend.full((2,), 5) == [5, 5]
    assert ListBackend.full((2, 2), 0) == [[0, 0], [0, 0]]
    assert ListBackend.concatenate([[1, 2], [3]]) == [1, 2, 3]
    assert ListBackend.flatten([[1, 2], [3, 4], [5, 6]]) == [1, 3, 5, 2, 4, 6]
    assert ListBackend.flatten([[[1, 2], [3]], [[4], [5, 6, 7]]]) == [[1, 2], [4], [3], [5, 6, 7]]
    assert ListBackend.pack_padded_sequence([[10, 11], [12, 13], [14, 15]],
                                            np.array([[True, True], [True, False], [False, True]])) == [10, 12, 11, 15]
    assert ListBackend.inf() == np.inf

    x = np.array([1.0, 4.0])
    y = np.array([3.0, 2.0])
    assert np.array_equal(ListBackend.maximum(x, y), np.array([3.0, 4.0]))
    assert np.array_equal(ListBackend.minimum(x, y), np.array([1.0, 2.0]))
    assert ListBackend.max(x, None) == 4.0
    assert ListBackend.min(x, None) == 1.0
    assert np.isclose(ListBackend.norm(np.array([3.0, 4.0])), 5.0)
    assert np.array_equal(ListBackend.logical_and(np.array([True, False]), np.array([True, True])),
                          np.array([True, False]))
    assert ListBackend.sum(x) == 5.0
    assert np.array_equal(ListBackend.stack([x, y], 0), np.stack([x, y], axis=0))

    assert ListBackend.size([[1, 2, 3], [4, 5, 6]]) == 6
    assert np.array_equal(ListBackend.squeeze([[1, 2, 3]]), np.array([1, 2, 3]))
    assert np.array_equal(ListBackend.atleast_2d([1, 2, 3]), np.array([[1, 2, 3]]))
    assert np.array_equal(ListBackend.repeat([1, 2], 2), np.array([1, 1, 2, 2]))
    assert np.array_equal(ListBackend.nonzero([0, 1, 0, 2]), np.array([1, 3]))
    assert np.array_equal(ListBackend.abs([-1, -2, 3]), np.array([1, 2, 3]))
    assert np.allclose(ListBackend.exp([0.0, 1.0]), np.array([1.0, np.e]))
    assert np.array_equal(ListBackend.sqrt([4.0, 9.0]), np.array([2.0, 3.0]))
    assert np.array_equal(ListBackend.clip([-5, 0, 5], -1, 1), np.array([-1, 0, 1]))
    assert np.array_equal(ListBackend.arange(0, 5), np.arange(5))

    with pytest.raises(ValueError):
        ListBackend.rand(2, device='cuda')
    with pytest.raises(ValueError):
        ListBackend.randint(0, 1, (1,), device='cuda')
    with pytest.raises(ValueError):
        ListBackend.arange(0, 1, device='cuda')

    np.random.seed(42)
    assert ListBackend.rand(2, 3).shape == (2, 3)
    randint_sample = ListBackend.randint(0, 5, (10,))
    assert randint_sample.shape == (10,) and np.all((randint_sample >= 0) & (randint_sample < 5))
    u = ListBackend.uniform(0.0, 1.0)
    assert 0.0 <= u <= 1.0
    assert ListBackend.multinomial(np.array([1.0, 0.0, 0.0])) == 0


def test_backend_ops_numpy():
    cond = np.array([True, False])
    assert np.array_equal(NumpyBackend.where(cond, np.array([1, 1]), np.array([2, 2])), np.array([1, 2]))
    assert NumpyBackend.size(np.ones((2, 3))) == 6
    assert NumpyBackend.inf() == np.inf
    assert NumpyBackend.max(np.array([1.0, 3.0, 2.0])) == 3.0
    assert NumpyBackend.min(np.array([1.0, 3.0, 2.0])) == 1.0
    assert np.isclose(NumpyBackend.norm(np.array([3.0, 4.0])), 5.0)
    assert np.array_equal(NumpyBackend.logical_and(np.array([True, False]), np.array([True, True])),
                          np.array([True, False]))
    assert NumpyBackend.sum(np.array([1.0, 2.0, 3.0])) == 6.0
    assert NumpyBackend.median(np.array([3.0, 1.0, 2.0])) == 2.0

    np.random.seed(42)
    u = NumpyBackend.uniform(0.0, 1.0)
    assert 0.0 <= u <= 1.0


def test_backend_ops_torch():
    cond = torch.tensor([True, False])
    assert torch.equal(TorchBackend.where(cond, torch.tensor([1, 1]), torch.tensor([2, 2])),
                       torch.tensor([1, 2]))
    assert TorchBackend.size(torch.ones(2, 3)) == 6
    assert TorchBackend.inf() == torch.inf
    assert TorchBackend.max(torch.tensor([1.0, 3.0, 2.0]), 0) == 3.0
    assert TorchBackend.min(torch.tensor([1.0, 3.0, 2.0]), 0) == 1.0
    assert torch.isclose(TorchBackend.norm(torch.tensor([3.0, 4.0])), torch.tensor(5.0))
    assert torch.equal(TorchBackend.logical_and(torch.tensor([True, False]), torch.tensor([True, True])),
                       torch.tensor([True, False]))
    assert TorchBackend.sum(torch.tensor([1.0, 2.0, 3.0])) == 6.0
    assert TorchBackend.median(torch.tensor([3.0, 1.0, 2.0])) == 2.0
    assert torch.equal(TorchBackend.squeeze(torch.ones(2, 1), 1), torch.ones(2))
    kept = torch.tensor([1, 2])
    assert TorchBackend.as_array(kept) is kept


def test_to_backend_dtype_numpy():
    assert NumpyBackend.to_backend_dtype(np.uint8) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(np.float32) == np.dtype('float32')
    assert NumpyBackend.to_backend_dtype(float) == np.dtype('float64')
    assert NumpyBackend.to_backend_dtype(int) == np.dtype('int64')
    assert NumpyBackend.to_backend_dtype(np.dtype('uint8')) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(torch.uint8) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(torch.float32) == np.dtype('float32')


def test_to_backend_dtype_torch():
    assert TorchBackend.to_backend_dtype(np.uint8) == torch.uint8
    assert TorchBackend.to_backend_dtype(np.float32) == torch.float32
    assert TorchBackend.to_backend_dtype(float) == torch.float32
    assert TorchBackend.to_backend_dtype(int) == torch.int64
    assert TorchBackend.to_backend_dtype(np.dtype('uint8')) == torch.uint8
    assert TorchBackend.to_backend_dtype(torch.uint8) == torch.uint8
    assert TorchBackend.to_backend_dtype(torch.float32) == torch.float32


def sequence_generator():
    list_n_steps = np.random.randint(2, 10, 100)
    list_n_envs = np.random.randint(2, 20, 100)
    list_n_dim = np.random.randint(1, 10, 100)

    for n_steps, n_envs, n_dim in zip(list_n_steps, list_n_envs, list_n_dim):
        lengths = np.random.randint(1, n_steps, size=(n_envs,))

        array = list()

        for d in range(n_dim):
            offset_dim = 10 * d

            array_dim = list()
            for e in range(n_envs):
                offset_env = 100 * e
                array_env_dim = offset_dim + offset_env + np.arange(0, n_steps)

                array_dim.append(array_env_dim)

            array_dim = np.stack(array_dim).T

            array.append(array_dim)

        array = np.stack(array, axis=-1).squeeze()

        mask = (np.arange(len(array))[:, None] < lengths[None, :])

        yield array, mask, lengths


def test_packed_2d_sequence_numpy():
    print('testing 2d sequence')
    array = np.arange(0, 100).reshape(20, 5, order='F')
    desired = np.concatenate([np.arange(0, 60), np.arange(60, 70), np.arange(80, 90)])
    mask = np.ones(100, dtype=bool).reshape(20, 5, order='F')

    mask[10:, 3:] = False

    print(mask)

    packed = NumpyBackend.pack_padded_sequence(array, mask)

    print('array')
    print(array)

    print('packed')
    print(packed)
    print('desired')
    print(desired)

    assert (packed == desired).all()


def test_pack_sequence_numpy():
    numpy.random.seed(42)

    for array, mask, lengths in sequence_generator():
        print('################################## Numpy')
        print('original')
        print(array)

        print('lengths')
        print(lengths)

        packed_array = NumpyBackend.pack_padded_sequence(array, mask)
        print('packed')
        print(packed_array)

        desired_array = np.concatenate([array[:l, i] for i, l in enumerate(lengths)])
        print('desired')
        print(desired_array)

        assert np.array_equal(desired_array, packed_array)


def test_packed_2d_sequence_torch():
    print('testing 2d sequence')
    array = torch.arange(0, 100).reshape(5, 20).T
    desired = torch.concatenate([torch.arange(0, 60), torch.arange(60, 70), torch.arange(80, 90)])
    mask = torch.ones(100, dtype=torch.bool).reshape(20, 5)

    mask[10:, 3:] = False
    print('mask')
    print(mask)

    packed = TorchBackend.pack_padded_sequence(array, mask)

    print('array')
    print(array)

    print('packed')
    print(packed)
    print('desired')
    print(desired)

    assert (packed == desired).all()


def test_pack_sequence_torch():
    numpy.random.seed(42)

    for array, mask, lengths in sequence_generator():
        torch_array = torch.as_tensor(array)
        mask = torch.as_tensor(mask)

        print('original')
        print(torch_array)

        print('lengths')
        print(lengths)

        packed_array = TorchBackend.pack_padded_sequence(torch_array, mask)
        print('packed')
        print(packed_array)

        desired_array = torch.concatenate([torch_array[:l, i] for i, l in enumerate(lengths)])
        print('desired')
        print(desired_array)

        assert torch.equal(packed_array, desired_array)
