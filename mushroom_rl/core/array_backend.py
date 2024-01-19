from collections import deque
import numpy as np
import torch

from mushroom_rl.utils.torch import TorchUtils


class ArrayBackend(object):

    @staticmethod
    def get_backend_name():
        raise NotImplementedError

    @staticmethod
    def get_array_backend(backend_name):
        assert type(backend_name) == str, f"Backend has to be string, not {type(backend_name).__name__}."
        if backend_name == 'numpy':
            return NumpyBackend
        elif backend_name == 'torch':
            return TorchBackend
        elif backend_name == 'list':
            return ListBackend
        else:
            raise ValueError(f"Unknown backend {backend_name}.")

    @staticmethod
    def get_array_backend_from(array):
        if isinstance(array, np.ndarray):
            return NumpyBackend
        elif isinstance(array, torch.Tensor):
            return TorchBackend
        elif isinstance(array, (list, deque)):
            return ListBackend
        else:
            raise ValueError(f"Unknown backend for type {type(array)}.")

    @classmethod
    def convert(cls, *arrays, to=None, backend=None):
        if to is None:
            to = cls.get_backend_name()
        if backend is None:
            backend = ArrayBackend.get_array_backend_from(arrays[0])
        if to == 'numpy':
            return backend.arrays_to_numpy(*arrays) if len(arrays) > 1 else backend.arrays_to_numpy(*arrays)[0]
        elif to == 'torch':
            return backend.arrays_to_torch(*arrays) if len(arrays) > 1 else backend.arrays_to_torch(*arrays)[0]
        else:
            return NotImplementedError

    @staticmethod
    def convert_to_backend(cls, array):
        raise NotImplementedError

    @classmethod
    def arrays_to_numpy(cls, *arrays):
        return tuple(cls.to_numpy(array) for array in arrays)

    @classmethod
    def arrays_to_torch(cls, *arrays):
        return tuple(cls.to_torch(array) for array in arrays)

    @classmethod
    def check_device(cls, device):
        if device is not None:
            raise ValueError(f"Device can not be set for {cls.get_backend_name()} backend.")

    @staticmethod
    def to_numpy(array):
        raise NotImplementedError

    @staticmethod
    def to_torch(array):
        raise NotImplementedError

    @classmethod
    def zeros(cls, *dims, dtype, device=None):
        raise NotImplementedError

    @classmethod
    def ones(cls, *dims, dtype, device=None):
        raise NotImplementedError

    @classmethod
    def zeros_like(cls, array, dtype, device=None):
        raise NotImplementedError

    @classmethod
    def ones_like(cls, array, dtype, device=None):
        raise NotImplementedError

    @staticmethod
    def concatenate(list_of_arrays, dim):
        raise NotImplementedError

    @staticmethod
    def where(cond, x=None, y=None):
        raise NotImplementedError

    @staticmethod
    def squeeze(array, dim):
        raise NotImplementedError

    @staticmethod
    def expand_dims(array, dim):
        raise NotImplementedError

    @staticmethod
    def size(arr):
        raise NotImplementedError

    @staticmethod
    def randint(low, high, size):
        raise NotImplementedError

    @staticmethod
    def arange(start, stop, step=1, dtype=None):
        raise NotImplementedError

    @staticmethod
    def abs(array):
        raise NotImplementedError

    @staticmethod
    def clip(array, min, max):
        raise NotImplementedError

    @staticmethod
    def copy(array):
        raise NotImplementedError

    @staticmethod
    def median(array):
        raise NotImplementedError

    @staticmethod
    def sqrt(array):
        raise NotImplementedError

    @staticmethod
    def from_list(array):
        raise NotImplementedError

    @staticmethod
    def pack_padded_sequence(array, mask):
        raise NotImplementedError


class NumpyBackend(ArrayBackend):
    @staticmethod
    def get_backend_name():
        return 'numpy'

    @staticmethod
    def to_numpy(array):
        return array

    @staticmethod
    def to_torch(array):
        return None if array is None else torch.from_numpy(array).to(TorchUtils.get_device())

    @staticmethod
    def convert_to_backend(cls, array):
        return cls.to_numpy(array)

    @classmethod
    def zeros(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return np.zeros(dims, dtype=dtype)

    @classmethod
    def ones(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return np.ones(dims, dtype=dtype)

    @classmethod
    def zeros_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return np.zeros_like(array, dtype=dtype)

    @classmethod
    def ones_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return np.ones_like(array, dtype=dtype)

    @staticmethod
    def concatenate(list_of_arrays, dim=0):
        return np.concatenate(list_of_arrays, axis=dim)

    @staticmethod
    def where(cond, x=None, y=None):
        assert (x is None) == (y is None), "Either both or neither of x and y should be given."
        if x is None:
            return np.where(cond)
        else:
            np.where(cond, x, y)

    @staticmethod
    def squeeze(array, dim=None):
        return np.squeeze(array, axis=dim)

    @staticmethod
    def expand_dims(array, dim):
        return np.expand_dims(array, axis=dim)

    @staticmethod
    def size(arr):
        return np.size(arr)

    @staticmethod
    def randint(low, high, size):
        assert type(size) == tuple
        return np.random.randint(low, high, size)

    @staticmethod
    def arange(start, stop, step=1, dtype=None):
        return np.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def abs(array):
        return np.abs(array)

    @staticmethod
    def clip(array, min, max):
        return np.clip(array, min, max)

    @staticmethod
    def copy(array):
        return array.copy()

    @staticmethod
    def median(array):
        return np.median(array)

    @staticmethod
    def sqrt(array):
        return np.sqrt(array)

    @staticmethod
    def from_list(array):
        return np.array(array)

    @staticmethod
    def pack_padded_sequence(array, mask):
        shape = array.shape

        new_shape = (shape[0] * shape[1],) + shape[2:]
        return array.reshape(new_shape, order='F')[mask.flatten(order='F')]


class TorchBackend(ArrayBackend):

    @staticmethod
    def get_backend_name():
        return 'torch'

    @staticmethod
    def to_numpy(array):
        return None if array is None else array.detach().cpu().numpy()

    @staticmethod
    def to_torch(array):
        return array

    @staticmethod
    def convert_to_backend(cls, array):
        return cls.to_torch(array)

    @classmethod
    def zeros(cls, *dims, dtype=torch.float32, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.zeros(*dims, dtype=dtype, device=device)

    @classmethod
    def ones(cls, *dims, dtype=torch.float32, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.ones(*dims, dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, array, dtype=torch.float32, device=None):
        device = array.device if device is None else device
        return torch.zeros_like(array, dtype=dtype, device=device)

    @classmethod
    def ones_like(cls, array, dtype=torch.float32, device=None):
        device = array.device if device is None else device
        return torch.ones_like(array, dtype=dtype, device=device)

    @staticmethod
    def concatenate(list_of_arrays, dim=0):
        return torch.concat(list_of_arrays, dim=dim)

    @staticmethod
    def where(cond, x=None, y=None):
        assert (x is None) == (y is None), "Either both or neither of x and y should be given."
        if x is None:
            return torch.where(cond)
        else:
            torch.where(cond, x, y)

    @staticmethod
    def squeeze(array, dim=None):
        if dim is None:
            return torch.squeeze(array)
        else:
            return torch.squeeze(array, dim=dim)

    @staticmethod
    def expand_dims(array, dim):
        return torch.unsqueeze(array, dim=dim)

    @staticmethod
    def size(arr):
        return torch.numel(arr)

    @staticmethod
    def randint(low, high, size):
        return torch.randint(low, high, size)

    @staticmethod
    def arange(start, stop, step=1, dtype=None):
        return torch.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def abs(array):
        return torch.abs(array)

    @staticmethod
    def clip(array, min, max):
        return torch.clip(array, min, max)

    @staticmethod
    def copy(array):
        return array.clone()

    @staticmethod
    def median(array):
        return array.median()

    @staticmethod
    def sqrt(array):
        return torch.sqrt(array)

    @staticmethod
    def from_list(array):
        if len(array) > 1 and isinstance(array[0], torch.Tensor):
            return torch.stack(array)
        else:
            return torch.tensor(array)

    @staticmethod
    def pack_padded_sequence(array, mask):
        shape = array.shape

        new_shape = (shape[0]*shape[1], ) + shape[2:]

        return array.transpose(0, 1).reshape(new_shape)[mask.transpose(0, 1).flatten()]


class ListBackend(ArrayBackend):

    @staticmethod
    def get_backend_name():
        return 'list'

    @staticmethod
    def to_numpy(array):
        return np.array(array)

    @staticmethod
    def to_torch(array):
        return None if array is None else torch.as_tensor(array, device=TorchUtils.get_device())

    @staticmethod
    def convert_to_backend(cls, array):
        return cls.to_numpy(array)

    @classmethod
    def zeros(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return np.zeros(dims, dtype=dtype)

    @classmethod
    def ones(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return np.ones(dims, dtype=dtype)

    @classmethod
    def zeros_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return np.zeros_like(array, dtype=dtype)

    @classmethod
    def ones_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return np.ones_like(array, dtype=dtype)

    @staticmethod
    def copy(array):
        return array.copy()

    @staticmethod
    def median(array):
        return np.median(array)

    @staticmethod
    def from_list(array):
        return array

    @staticmethod
    def pack_padded_sequence(array, mask):
        return NumpyBackend.pack_padded_sequence(array, np.array(mask))
