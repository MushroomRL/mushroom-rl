import numpy
import torch


class DataConversion(object):
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


class NumpyConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return array

    @staticmethod
    def to_torch(array):
        return torch.from_numpy(array)


class TorchConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return array.detach().cpu().numpy()

    @staticmethod
    def to_torch(array):
        return array


class ListConversion(DataConversion):
    @staticmethod
    def to_numpy(array):
        return numpy.array(array)

    @staticmethod
    def to_torch(array):
        return torch.as_tensor(array)




