import numpy as np
import torch


def make_array(backend, data, device=None):
    if backend == 'torch':
        return torch.tensor(data, device=device)
    return np.array(data)


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, (list, tuple)):
        return np.array([to_numpy(item) for item in array])
    return np.asarray(array)


def assert_backend(array, backend, device=None):
    if backend == 'torch':
        assert isinstance(array, torch.Tensor)
        assert array.device.type == device
    elif backend == 'numpy':
        assert isinstance(array, np.ndarray)
    else:
        assert isinstance(array, list)


def assert_equal(array, expected, backend=None, device=None):
    if backend is not None:
        assert_backend(array, backend, device)
    assert np.array_equal(to_numpy(array), to_numpy(expected))
