import numpy as np
import torch

from mushroom_rl.utils.torch import TorchUtils


class DataConversion(object):
    @staticmethod
    def get_converter(backend):
        if backend == 'numpy':
            return NumpyConversion
        elif backend == 'torch':
            return TorchConversion
        else:
            return ListConversion

    @classmethod
    def convert(cls, *arrays, to='numpy'):
        if to == 'numpy':
            return cls.arrays_to_numpy(*arrays)
        elif to == 'torch':
            return cls.arrays_to_torch(*arrays)
        else:
            return NotImplementedError

    @classmethod
    def arrays_to_numpy(cls, *arrays):
        return (cls.to_numpy(array) for array in arrays)

    @classmethod
    def arrays_to_torch(cls, *arrays):
        return (cls.to_torch(array) for array in arrays)

    @staticmethod
    def to_numpy(array):
        return NotImplementedError

    @staticmethod
    def to_torch(array):
        raise NotImplementedError

    @staticmethod
    def to_backend_array(cls, array):
        raise NotImplementedError

    @staticmethod
    def zeros(*dims, dtype):
        raise NotImplementedError

    @staticmethod
    def ones(*dims, dtype):
        raise NotImplementedError

    @staticmethod
    def copy(array):
        raise NotImplementedError


class NumpyConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return array

    @staticmethod
    def to_torch(array):
        return None if array is None else torch.from_numpy(array).to(TorchUtils.get_device())

    @staticmethod
    def to_backend_array(cls, array):
        return cls.to_numpy(array)

    @staticmethod
    def zeros(*dims, dtype=float):
        return np.zeros(dims, dtype=dtype)

    @staticmethod
    def ones(*dims, dtype=float):
        return np.ones(dims, dtype=dtype)

    @staticmethod
    def copy(array):
        return array.copy()


class TorchConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return None if array is None else array.detach().cpu().numpy()

    @staticmethod
    def to_torch(array):
        return array

    @staticmethod
    def to_backend_array(cls, array):
        return cls.to_torch(array)

    @staticmethod
    def zeros(*dims, dtype=torch.float32):
        return torch.zeros(*dims, dtype=dtype, device=TorchUtils.get_device())

    @staticmethod
    def ones(*dims, dtype=torch.float32):
        return torch.ones(*dims, dtype=dtype, device=TorchUtils.get_device())

    @staticmethod
    def copy(array):
        return array.clone()


class ListConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return np.array(array)

    @staticmethod
    def to_torch(array):
        return None if array is None else torch.as_tensor(array, device=TorchUtils.get_device())

    @staticmethod
    def to_backend_array(cls, array):
        return cls.to_numpy(array)

    @staticmethod
    def zeros(*dims, dtype=float):
        return np.zeros(dims, dtype=float)

    @staticmethod
    def ones(*dims, dtype=float):
        return np.ones(dims, dtype=float)

    @staticmethod
    def copy(array):
        return array.copy()




