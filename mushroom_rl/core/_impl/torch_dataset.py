import numpy as np
import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.utils.torch_utils import TorchUtils


class TorchDataset(MushroomObject):
    """
    Preallocated storage using torch tensors. Holds an ordered list of equal-length columns (one tensor per column),
    each independently shaped and typed, plus a shared length counter.

    """
    def __init__(self, shapes, dtypes, device=None, n_envs=None):
        self._device = TorchUtils.get_device(device)
        self._arrays = [torch.empty(shape, dtype=dtype, device=self._device)
                        for shape, dtype in zip(shapes, dtypes)]
        self._n_envs = n_envs
        self._len = 0

        self._add_all_save_attr()

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return tuple(array[index] for array in self._arrays)

    def __add__(self, other):
        result = self.create_new_instance(self)

        result._arrays = [torch.concatenate((array[:self._len], other_array[:len(other)]))
                          for array, other_array in zip(self._arrays, other._arrays)]
        result._len = self._len + len(other)

        return result

    def append(self, *values):
        i = self._len
        for array, value in zip(self._arrays, values):
            array[i] = value if self._n_envs is None else value[:self._n_envs]
        self._len += 1

    def append_batch(self, other):
        n = len(other)
        i = self._len
        assert i + n <= self.capacity
        for array, column in zip(self._arrays, other.data):
            array[i:i + n] = column
        self._len += n

    def clear(self):
        self._arrays = [torch.empty_like(array) for array in self._arrays]
        self._len = 0

    def reserve(self, capacity):
        if capacity > self.capacity:
            new_arrays = list()
            for array in self._arrays:
                buffer = torch.empty((capacity,) + array.shape[1:], dtype=array.dtype, device=array.device)
                buffer[:self._len] = array[:self._len]
                new_arrays.append(buffer)
            self._arrays = new_arrays

    def get_view(self, index, copy=False):
        view = self.create_new_instance(self)

        if copy:
            view._arrays = list()
            new_len = 0
            for array in self._arrays:
                buffer = torch.empty_like(array)
                sliced = array[:self._len][index, ...]
                new_len = sliced.shape[0]
                buffer[:new_len] = sliced
                view._arrays.append(buffer)
            view._len = new_len
        else:
            view._arrays = [array[:self._len][index, ...] for array in self._arrays]
            view._len = view._arrays[0].shape[0]

        return view

    def column(self, index=None):
        if index is None:
            assert self.n_columns == 1
            index = 0
        return self._arrays[index][:self._len]

    def n_episodes(self, last_index):
        last = self.column(last_index)

        if len(last) == 0:
            return 0

        n_episodes = last.sum()

        if not last[-1]:
            n_episodes += 1

        return n_episodes

    @classmethod
    def create_new_instance(cls, dataset=None):
        """
        Creates an empty instance of the dataset and populates essential data structures.

        Args:
            dataset (TorchDataset, None): a template dataset to be used to create the new instance.

        Returns:
            A new empty instance of the dataset.

        """
        new_dataset = cls.__new__(cls)

        new_dataset._device = dataset._device if dataset is not None else None
        new_dataset._arrays = None
        new_dataset._n_envs = dataset._n_envs if dataset is not None else None
        new_dataset._len = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, arrays, device=None):
        dataset = cls.create_new_instance()

        dataset._device = TorchUtils.get_device(device)
        dataset._arrays = [dataset._to_tensor(array) for array in arrays]
        dataset._len = len(dataset._arrays[0])

        return dataset

    @property
    def data(self):
        return [array[:self._len] for array in self._arrays]

    @property
    def n_columns(self):
        return len(self._arrays)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def capacity(self):
        return self._arrays[0].shape[0]

    def _to_tensor(self, array):
        if isinstance(array, torch.Tensor):
            return array.to(device=self._device)
        return torch.from_numpy(np.asarray(array)).to(device=self._device)

    def _add_all_save_attr(self):
        self._add_save_attr(
            _device='primitive',
            _arrays='torch',
            _n_envs='primitive',
            _len='primitive'
        )
