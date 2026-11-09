import numpy as np
import pytest
import torch

from helper.arrays import assert_equal, make_array, to_numpy
from helper.cases import backend_devices

from mushroom_rl.core.array_backend import ArrayBackend, NumpyBackend, TorchBackend, ListBackend


def test_lookup():
    assert ArrayBackend.get_array_backend('numpy') is NumpyBackend
    assert ArrayBackend.get_array_backend('torch') is TorchBackend
    assert ArrayBackend.get_array_backend('list') is ListBackend
    assert ArrayBackend.get_array_backend_from(np.zeros(2)) is NumpyBackend
    assert ArrayBackend.get_array_backend_from(torch.zeros(2)) is TorchBackend
    assert ArrayBackend.get_array_backend_from([1, 2]) is ListBackend
    assert NumpyBackend.supports_devices() is False
    assert ListBackend.supports_devices() is False
    assert TorchBackend.supports_devices() is True

    with pytest.raises(AssertionError):
        ArrayBackend.get_array_backend(1)
    with pytest.raises(ValueError):
        ArrayBackend.get_array_backend('unknown')
    with pytest.raises(ValueError):
        ArrayBackend.get_array_backend_from(1)


def test_abstract():
    calls = [
        lambda: ArrayBackend.get_backend_name(),
        lambda: ArrayBackend.get_backend_serialization(),
        lambda: ArrayBackend.get_container_serialization(),
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
        lambda: ArrayBackend.argsort(None),
        lambda: ArrayBackend.sqrt(None),
        lambda: ArrayBackend.from_list(None),
        lambda: ArrayBackend.pack_padded_sequence(None, None),
        lambda: ArrayBackend.masked_select(None, None),
        lambda: ArrayBackend.masked_assign(None, None, None),
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
        lambda: ArrayBackend.diag(None),
        lambda: ArrayBackend.diagonal(None),
        lambda: ArrayBackend.cholesky(None),
        lambda: ArrayBackend.inv(None),
        lambda: ArrayBackend.logical_and(None, None),
        lambda: ArrayBackend.sum(None),
        lambda: ArrayBackend.stack([], 0),
        lambda: ArrayBackend.to_backend_dtype(float),
        lambda: ArrayBackend.to_list(None),
        lambda: ArrayBackend.as_array(None),
        lambda: ArrayBackend.masked_init(None, None),
        lambda: ArrayBackend.cumsum(None),
    ]

    for call in calls:
        with pytest.raises(NotImplementedError):
            call()


def test_list_backend():
    assert ListBackend.get_backend_name() == 'list'
    assert ListBackend.get_backend_serialization() == 'numpy'
    assert ListBackend.get_container_serialization() == 'pickle'

    assert np.array_equal(ListBackend.to_numpy([1, 2, 3]), np.array([1, 2, 3]))
    assert torch.equal(ListBackend.to_torch([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert ListBackend.to_backend_dtype(np.float32) == np.float32
    raw_object = [1, 2]
    assert ListBackend.convert_to_backend(ListBackend, raw_object) is raw_object
    assert np.array_equal(ListBackend.as_array([1, 2, 3]), np.array([1, 2, 3]))

    assert np.array_equal(ListBackend.zeros(2, 3, dtype=float), np.zeros((2, 3)))
    assert np.array_equal(ListBackend.ones(2, 3, dtype=float), np.ones((2, 3)))
    assert np.array_equal(ListBackend.zeros_like(np.ones(3)), np.zeros(3))
    assert np.array_equal(ListBackend.ones_like(np.zeros(3)), np.ones(3))

    array = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(ListBackend.copy(array), array)

    nested = [[1, 2], [3, 4]]
    nested_copy = ListBackend.copy(nested)
    nested_copy[0].append(99)
    assert nested == [[1, 2], [3, 4]]

    mask = np.array([False, True, False, True])
    values = [[1, 2], [3, 4]]
    scattered = ListBackend.masked_init(mask, values)
    assert scattered == [None, [1, 2], None, [3, 4]]
    scattered[1].append(99)
    assert values == [[1, 2], [3, 4]]

    assert ListBackend.from_list(array) is array
    assert ListBackend.empty((3,)) == [None, None, None]
    assert ListBackend.empty((2, 2)) == [[None, None], [None, None]]
    assert ListBackend.none() is None
    assert ListBackend.shape([[1, 2], [3, 4]]) == (2, 2)
    assert ListBackend.full((2,), 5) == [5, 5]
    assert ListBackend.full((2, 2), 0) == [[0, 0], [0, 0]]
    assert ListBackend.flatten([[1, 2], [3, 4], [5, 6]]) == [1, 3, 5, 2, 4, 6]
    assert ListBackend.flatten([[[1, 2], [3]], [[4], [5, 6, 7]]]) == [[1, 2], [4], [3], [5, 6, 7]]
    assert ListBackend.pack_padded_sequence([[10, 11], [12, 13], [14, 15]],
                                            np.array([[True, True], [True, False], [False, True]])) == [10, 12, 11, 15]
    assert np.array_equal(ListBackend.nonzero([0, 1, 0, 2]), np.array([1, 3]))

    selection_mask = np.array([True, False, True])
    assert ListBackend.masked_select([[1, 2], [3, 4], [5, 6]], selection_mask) == [[1, 2], [5, 6]]

    target = [[1, 2], [3, 4], [5, 6]]
    written = [[7, 8], [9, 10]]
    ListBackend.masked_assign(target, selection_mask, written)
    assert target == [[7, 8], [3, 4], [9, 10]]
    target[0].append(99)
    assert written == [[7, 8], [9, 10]]

    ListBackend.masked_assign(target, np.array([False, False, False]), [])
    assert target == [[7, 8, 99], [3, 4], [9, 10]]

    assert ListBackend.concatenate([[1, 2], [3]]) == [1, 2, 3]
    assert ListBackend.concatenate([[1, 2], [3]], dim=0) == [1, 2, 3]
    assert np.array_equal(ListBackend.concatenate_arrays([np.array([1, 2]), np.array([3])]), [1, 2, 3])
    assert np.array_equal(ListBackend.cumsum([1, 2, 3]), [1, 3, 6])
    first = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    second = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    concatenated = ListBackend.concatenate([first, second], dim=1)
    assert np.array_equal(concatenated, NumpyBackend.concatenate([np.array(first), np.array(second)], dim=1))
    assert np.array_equal(concatenated, np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]))

    data = [np.array([0.5, 1.5]), np.array([2.5, 3.5])]
    converted = ListBackend.to_torch(data)
    assert converted.dtype == torch.float32
    assert converted.dtype == NumpyBackend.to_torch(np.array(data)).dtype
    assert torch.equal(converted, torch.tensor([[0.5, 1.5], [2.5, 3.5]]))
    assert ListBackend.to_torch([[1, 2], [3, 4]]).dtype == torch.int64
    assert ListBackend.to_torch([True, False]).dtype == torch.bool
    assert ListBackend.to_torch(None) is None

    with pytest.raises(ValueError):
        ListBackend.zeros(2, dtype=float, device='cuda')
    with pytest.raises(ValueError):
        ListBackend.rand(2, device='cuda')
    with pytest.raises(ValueError):
        ListBackend.randint(0, 1, (1,), device='cuda')
    with pytest.raises(ValueError):
        ListBackend.arange(0, 1, device='cuda')


@pytest.mark.parametrize('backend, device', backend_devices())
def test_math(backend, device):
    ops = ArrayBackend.get_array_backend(backend)
    out = None if backend == 'list' else backend

    def array(data):
        return data if backend == 'list' else make_array(backend, data, device)

    x, y = array([1.0, 4.0]), array([3.0, 2.0])
    matrix = array([[4.0, 2.0], [2.0, 3.0]])

    assert_equal(ops.where(array([True, False]), array([1, 1]), array([2, 2])), [1, 2], out, device)
    assert_equal(ops.nonzero(array([0, 1, 0, 2])), [1, 3], out, device)
    assert_equal(ops.nonzero(array([0, 1, 0])), [1], out, device)
    assert ops.size(array([[1, 2, 3], [4, 5, 6]])) == 6
    assert ops.inf() == np.inf
    assert ops.to_torch(None) is None
    assert np.shape(ops.empty((2, 3), device=device)) == (2, 3)
    assert ops.max(array([1.0, 3.0, 2.0])) == 3.0
    assert ops.min(array([1.0, 3.0, 2.0])) == 1.0
    assert_equal(ops.max(matrix, 0), [4.0, 3.0], out, device)
    assert_equal(ops.min(matrix, 0), [2.0, 2.0], out, device)
    assert_equal(ops.maximum(x, y), [3.0, 4.0], out, device)
    assert_equal(ops.minimum(x, y), [1.0, 2.0], out, device)
    assert np.isclose(to_numpy(ops.norm(array([3.0, 4.0]))), 5.0)
    assert ops.sum(x) == 5.0
    assert ops.median(array([3.0, 1.0, 2.0])) == 2.0
    assert_equal(ops.argsort(array([3, 1, 2, 1])), [1, 3, 2, 0], out, device)
    assert_equal(ops.diag(array([1.0, 2.0])), [[1.0, 0.0], [0.0, 2.0]], out, device)
    assert_equal(ops.diagonal(matrix), [4.0, 3.0], out, device)
    assert np.allclose(to_numpy(ops.cholesky(matrix)), [[2.0, 0.0], [1.0, np.sqrt(2.0)]])
    assert np.allclose(to_numpy(ops.inv(matrix)), [[0.375, -0.25], [-0.25, 0.5]])
    assert_equal(ops.logical_and(array([True, False]), array([True, True])), [True, False], out, device)
    assert_equal(ops.stack([x, y], 0), [[1.0, 4.0], [3.0, 2.0]], out, device)
    assert_equal(ops.squeeze(array([[1, 2, 3]])), [1, 2, 3], out, device)
    assert_equal(ops.squeeze(array([[1], [2]]), 1), [1, 2], out, device)
    assert_equal(ops.atleast_2d(array([1, 2, 3])), [[1, 2, 3]], out, device)
    assert_equal(ops.repeat(array([1, 2]), 2), [1, 1, 2, 2], out, device)
    assert_equal(ops.abs(array([-1, -2, 3])), [1, 2, 3], out, device)
    assert np.allclose(to_numpy(ops.exp(array([0.0, 1.0]))), [1.0, np.e])
    assert_equal(ops.sqrt(array([4.0, 9.0])), [2.0, 3.0], out, device)
    assert_equal(ops.clip(array([-5, 0, 5]), -1, 1), [-1, 0, 1], out, device)
    assert_equal(ops.arange(0, 5, device=device), [0, 1, 2, 3, 4], out, device)

    if backend != 'list':
        kept = array([1, 2])
        assert ops.as_array(kept, device=device) is kept

        selection_mask = array([True, False, True])
        values = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert_equal(ops.masked_select(values, selection_mask), [[1.0, 2.0], [5.0, 6.0]], out, device)
        ops.masked_assign(values, selection_mask, array([[7.0, 8.0], [9.0, 10.0]]))
        assert_equal(values, [[7.0, 8.0], [3.0, 4.0], [9.0, 10.0]], out, device)


@pytest.mark.parametrize('backend, device', backend_devices())
def test_random(backend, device):
    ops = ArrayBackend.get_array_backend(backend)

    def draws():
        if backend == 'torch':
            torch.manual_seed(42)
            return (torch.rand((2, 3), device=device), torch.randint(0, 5, (10,), device=device),
                    torch.multinomial(torch.tensor([.2, .8], device=device), 1),
                    dict(cpu=2.5392632484436035, cuda=3.459198474884033)[device])
        np.random.seed(42)
        return (np.random.rand(2, 3), np.random.randint(0, 5, (10,)), np.array([np.random.choice(2, p=[.2, .8])]),
                np.random.uniform(2.0, 4.0))

    rand, randint, multinomial, uniform = draws()
    probabilities = make_array(backend, [.2, .8], device)
    if backend == 'torch':
        torch.manual_seed(42)
    else:
        np.random.seed(42)

    assert_equal(ops.rand(2, 3, device=device), rand)
    assert_equal(ops.randint(0, 5, (10,), device=device), randint)
    assert_equal(ops.multinomial(probabilities), multinomial)
    assert_equal(ops.uniform(2.0, 4.0, device=device), uniform)
    assert ListBackend.multinomial(np.array([1.0, 0.0, 0.0])) == 0


def test_dtype():
    assert NumpyBackend.to_backend_dtype(np.uint8) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(np.float32) == np.dtype('float32')
    assert NumpyBackend.to_backend_dtype(float) == np.dtype('float64')
    assert NumpyBackend.to_backend_dtype(int) == np.dtype('int64')
    assert NumpyBackend.to_backend_dtype(np.dtype('uint8')) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(torch.uint8) == np.dtype('uint8')
    assert NumpyBackend.to_backend_dtype(torch.float32) == np.dtype('float32')

    assert TorchBackend.to_backend_dtype(np.uint8) == torch.uint8
    assert TorchBackend.to_backend_dtype(np.float32) == torch.float32
    assert TorchBackend.to_backend_dtype(float) == torch.float32
    assert TorchBackend.to_backend_dtype(int) == torch.int64
    assert TorchBackend.to_backend_dtype(np.dtype('uint8')) == torch.uint8
    assert TorchBackend.to_backend_dtype(torch.uint8) == torch.uint8
    assert TorchBackend.to_backend_dtype(torch.float32) == torch.float32


def padded_sequences():
    list_n_steps = np.random.randint(2, 10, 100)
    list_n_envs = np.random.randint(2, 20, 100)
    list_n_dim = np.random.randint(1, 10, 100)

    for n_steps, n_envs, n_dim in zip(list_n_steps, list_n_envs, list_n_dim):
        lengths = np.random.randint(1, n_steps, size=(n_envs,))
        array = np.stack([np.stack([10 * d + 100 * e + np.arange(0, n_steps) for e in range(n_envs)]).T
                          for d in range(n_dim)], axis=-1).squeeze()
        mask = (np.arange(len(array))[:, None] < lengths[None, :])

        yield array, mask, lengths


@pytest.mark.parametrize('backend, device', backend_devices('numpy', 'torch'))
def test_pack_padded_sequence(backend, device):
    ops = ArrayBackend.get_array_backend(backend)
    mask = np.ones(100, dtype=bool).reshape(20, 5, order='F')
    mask[10:, 3:] = False
    array = np.arange(0, 100).reshape(20, 5, order='F')
    desired = np.concatenate([np.arange(0, 60), np.arange(60, 70), np.arange(80, 90)])

    packed = ops.pack_padded_sequence(make_array(backend, array, device), make_array(backend, mask, device))

    assert_equal(packed, desired, backend, device)

    np.random.seed(42)
    for array, mask, lengths in padded_sequences():
        packed = ops.pack_padded_sequence(make_array(backend, array, device), make_array(backend, mask, device))
        assert_equal(packed, np.concatenate([array[:length, i] for i, length in enumerate(lengths)]), backend, device)


def test_convert_mask():
    numpy_mask = np.array([True, False, True])
    torch_mask = torch.tensor([True, False, True])

    assert torch.equal(TorchBackend.convert_mask(numpy_mask), torch_mask)
    assert np.array_equal(NumpyBackend.convert_mask(torch_mask), numpy_mask)

    converted = ListBackend.convert_mask(torch_mask)
    assert isinstance(converted, np.ndarray)
    assert np.array_equal(converted, numpy_mask)

    first, second = ListBackend.convert_mask(torch_mask, torch_mask)
    assert isinstance(first, np.ndarray)
    assert isinstance(second, np.ndarray)

    assert isinstance(ArrayBackend.convert_mask(numpy_mask, to='list'), np.ndarray)
    assert np.array_equal(ListBackend.convert_to_backend_mask(TorchBackend, torch_mask), numpy_mask)
