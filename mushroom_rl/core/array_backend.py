import copy
from collections import deque
import numpy as np
import torch

from mushroom_rl.utils.torch_utils import TorchUtils


class ArrayBackend(object):
    """
    Interface for the array backends used across MushroomRL. A backend abstracts the array type used to store
    and manipulate data (states, actions, rewards, ...) so that the same code can run on different array
    libraries. Three backends are provided: :class:`NumpyBackend`, :class:`TorchBackend` and
    :class:`ListBackend`, selected by name through :meth:`get_array_backend`.

    """
    @staticmethod
    def get_backend_name():
        """
        Returns:
            The name of the backend (``'numpy'``, ``'torch'`` or ``'list'``).

        """
        raise NotImplementedError

    @staticmethod
    def get_backend_serialization():
        """
        Returns:
            The name of the ``MushroomObject`` save method (see ``_add_save_attr``) to use for attributes
            stored in this backend's array type, i.e. ``'numpy'`` or ``'torch'``. Backends with no dedicated
            save method (e.g. :class:`ListBackend`) return the name of another backend able to serialize
            their data instead (``'numpy'``).

        """
        raise NotImplementedError

    @staticmethod
    def get_array_backend(backend_name):
        """
        Args:
            backend_name (str): name of the backend, one of ``'numpy'``, ``'torch'`` or ``'list'``.

        Returns:
            The :class:`ArrayBackend` subclass registered under ``backend_name``.

        """
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
        """
        Args:
            array (object): an array produced by one of the supported backends (a NumPy ``ndarray``, a PyTorch
                ``Tensor``, or a ``list``/``deque``).

        Returns:
            The :class:`ArrayBackend` subclass handling the type of ``array``.

        """
        if isinstance(array, np.ndarray):
            return NumpyBackend
        elif isinstance(array, torch.Tensor):
            return TorchBackend
        elif isinstance(array, (list, deque)):
            return ListBackend
        else:
            raise ValueError(f"Unknown backend for type {type(array)}.")

    @classmethod
    def check_device(cls, device):
        """
        Args:
            device: device requested by the caller.

        Raises:
            ValueError: if ``device`` is not ``None``. Only backends that support devices (i.e.
                :class:`TorchBackend`) override this check to accept a non-``None`` device.

        """
        if device is not None:
            raise ValueError(f"Device can not be set for {cls.get_backend_name()} backend.")

    @classmethod
    def convert(cls, *arrays, to=None, backend=None):
        """
        Convert one or more arrays from their current backend to another one.

        Args:
            *arrays: one or more arrays to convert;
            to (str, None): name of the destination backend. If ``None``, the backend calling this method
                (``cls``) is used;
            backend (ArrayBackend, None): backend of the input arrays. If ``None``, it is autodetected from
                the first element of ``arrays``.

        Returns:
            The converted array, or a tuple of converted arrays if more than one was passed in ``arrays``.

        """
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
    def convert_to_backend(backend, array):
        """
        Convert a single array from another backend into this backend's native array type. Unlike
        :meth:`convert`, this is a static method called on the destination backend, taking the source backend
        as an explicit positional argument, e.g. ``TorchBackend.convert_to_backend(NumpyBackend, array)``
        converts a NumPy ``array`` into a PyTorch ``Tensor``.

        Args:
            backend (ArrayBackend): the backend ``array`` currently belongs to;
            array: the array to convert.

        Returns:
            ``array`` converted into this backend's native format.

        """
        raise NotImplementedError

    @classmethod
    def arrays_to_numpy(cls, *arrays):
        """
        Args:
            *arrays: one or more arrays in this backend's format.

        Returns:
            A tuple with the arrays converted to NumPy ``ndarray``.

        """
        return tuple(cls.to_numpy(array) for array in arrays)

    @classmethod
    def arrays_to_torch(cls, *arrays):
        """
        Args:
            *arrays: one or more arrays in this backend's format.

        Returns:
            A tuple with the arrays converted to PyTorch ``Tensor``.

        """
        return tuple(cls.to_torch(array) for array in arrays)

    @classmethod
    def arrays_to_list(cls, *arrays):
        """
        Args:
            *arrays: one or more arrays in this backend's format.

        Returns:
            A tuple with the arrays converted to plain Python ``list``.

        """
        return tuple(cls.to_list(array) for array in arrays)

    @staticmethod
    def to_numpy(array):
        """
        Args:
            array: an array in this backend's format.

        Returns:
            ``array`` converted to a NumPy ``ndarray``.

        """
        raise NotImplementedError

    @staticmethod
    def to_torch(array):
        """
        Args:
            array: an array in this backend's format.

        Returns:
            ``array`` converted to a PyTorch ``Tensor``.

        """
        raise NotImplementedError

    @staticmethod
    def to_list(array):
        """
        Args:
            array: an array in this backend's format.

        Returns:
            ``array`` converted to a plain Python ``list``.

        """
        raise NotImplementedError

    @staticmethod
    def as_array(array):
        """
        Cast ``array`` to this backend's native array type without changing its backend, materializing it if
        needed (e.g. wrapping a plain Python list into a NumPy/PyTorch array).

        Args:
            array: an array-like object.

        Returns:
            ``array`` as a native object of this backend.

        """
        raise NotImplementedError

    @staticmethod
    def from_list(array):
        """
        Build a backend array from a plain Python list (the inverse of :meth:`to_list`).

        Args:
            array (list): a plain Python list.

        Returns:
            ``array`` converted to this backend's native array type.

        """
        raise NotImplementedError

    @staticmethod
    def to_backend_dtype(dtype):
        """
        Args:
            dtype: a dtype specification, either native to this backend or to another supported backend (e.g.
                a NumPy ``dtype`` or a PyTorch ``dtype``).

        Returns:
            ``dtype`` converted to this backend's native dtype representation.

        """
        raise NotImplementedError

    @staticmethod
    def empty(shape, device=None):
        """
        Args:
            shape: shape of the array to create;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            A new, uninitialized array of the given ``shape``.

        """
        raise NotImplementedError

    @staticmethod
    def full(shape, value):
        """
        Args:
            shape: shape of the array to create;
            value: fill value.

        Returns:
            A new array of the given ``shape``, filled with ``value``.

        """
        raise NotImplementedError

    @staticmethod
    def concatenate(list_of_arrays, dim):
        """
        Args:
            list_of_arrays: a list of arrays to concatenate;
            dim: dimension along which the arrays are concatenated.

        Returns:
            The arrays in ``list_of_arrays`` concatenated along ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def flatten(array):
        """
        Merge the first two axes of an array into a single leading axis, grouping the result by the second
        axis and ordering it by the first axis within each group (i.e. all entries with index 0 along the
        second axis, in order of their index along the first axis, then all entries with index 1 along the
        second axis, ...). This differs from a plain row-major reshape, which would order elements by the
        first axis first.

        Args:
            array: an array of shape ``(A, B, ...)``.

        Returns:
            ``array`` reshaped to ``(A * B, ...)``.

        """
        raise NotImplementedError

    @staticmethod
    def pack_padded_sequence(array, mask):
        """
        Select the entries of an array marked by a boolean mask over its first two axes, and concatenate them
        into a single leading axis, grouped by the second axis and ordered by the first axis within each
        group (i.e. all selected entries with index 0 along the second axis, in order of their index along
        the first axis, then all selected entries with index 1 along the second axis, ...).

        Args:
            array: an array of shape ``(A, B, ...)``;
            mask: a boolean array of shape ``(A, B)`` marking the entries of ``array`` to keep.

        Returns:
            The entries of ``array`` selected by ``mask``, concatenated in the order described above.

        """
        raise NotImplementedError

    @classmethod
    def zeros(cls, *dims, dtype, device=None):
        """
        Args:
            *dims: shape of the array to create;
            dtype: data type of the array;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            A new array of shape ``dims`` filled with zeros.

        """
        raise NotImplementedError

    @classmethod
    def ones(cls, *dims, dtype, device=None):
        """
        Args:
            *dims: shape of the array to create;
            dtype: data type of the array;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            A new array of shape ``dims`` filled with ones.

        """
        raise NotImplementedError

    @classmethod
    def zeros_like(cls, array, dtype, device=None):
        """
        Args:
            array: array whose shape is used for the new array;
            dtype: data type of the array;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            A new array with the same shape as ``array``, filled with zeros.

        """
        raise NotImplementedError

    @classmethod
    def ones_like(cls, array, dtype, device=None):
        """
        Args:
            array: array whose shape is used for the new array;
            dtype: data type of the array;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            A new array with the same shape as ``array``, filled with ones.

        """
        raise NotImplementedError

    @staticmethod
    def masked_init(mask, values):
        """
        Build an array of shape ``(len(mask), *values.shape[1:])`` where the entries selected by ``mask`` are
        filled, in order, with ``values``, and the remaining entries are left uninitialized.

        Args:
            mask: a boolean array of shape ``(N,)``;
            values: an array of shape ``(M, ...)``, with ``M`` equal to the number of ``True`` entries in
                ``mask``.

        Returns:
            An array of shape ``(N, ...)`` with ``values`` scattered at the positions where ``mask`` is
            ``True``.

        """
        raise NotImplementedError

    @staticmethod
    def shape(array):
        """
        Args:
            array: an array.

        Returns:
            The shape of ``array``, as a tuple.

        """
        raise NotImplementedError

    @staticmethod
    def size(arr):
        """
        Args:
            arr: an array.

        Returns:
            The total number of elements in ``arr``.

        """
        raise NotImplementedError

    @staticmethod
    def none():
        """
        Returns:
            This backend's representation of a missing value (``nan`` for :class:`NumpyBackend` and
            :class:`TorchBackend`, ``None`` for :class:`ListBackend`).

        """
        raise NotImplementedError

    @staticmethod
    def inf():
        """
        Returns:
            This backend's representation of positive infinity.

        """
        raise NotImplementedError

    @staticmethod
    def copy(array):
        """
        Args:
            array: an array.

        Returns:
            An independent copy of ``array``. For :class:`NumpyBackend`/:class:`TorchBackend` this is a
            shallow copy of the underlying numeric buffer, which is sufficient since they only ever hold
            regular numeric data. :class:`ListBackend` performs a deep copy instead, since it can hold
            nested/ragged Python containers whose inner elements a shallow copy would still alias.

        """
        raise NotImplementedError

    @staticmethod
    def squeeze(array, dim):
        """
        Args:
            array: an array;
            dim: dimension(s) to remove, must have size 1. If ``None``, all size-1 dimensions are removed.

        Returns:
            ``array`` with the size-1 dimension(s) removed.

        """
        raise NotImplementedError

    @staticmethod
    def expand_dims(array, dim):
        """
        Args:
            array: an array;
            dim: position where the new axis is inserted.

        Returns:
            ``array`` with a new size-1 axis inserted at position ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def atleast_2d(array):
        """
        Args:
            array: an array.

        Returns:
            ``array`` reshaped so that it has at least two dimensions.

        """
        raise NotImplementedError

    @staticmethod
    def repeat(array, repeats):
        """
        Args:
            array: an array;
            repeats: number of times each element is repeated.

        Returns:
            ``array`` with each element repeated ``repeats`` times.

        """
        raise NotImplementedError

    @staticmethod
    def stack(lst, dim):
        """
        Args:
            lst: a list of arrays with the same shape;
            dim: dimension along which the arrays are stacked.

        Returns:
            The arrays in ``lst`` stacked along a new dimension ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def where(cond, x=None, y=None):
        """
        Args:
            cond: a boolean array/condition;
            x (None): array of values to select where ``cond`` is ``True``;
            y (None): array of values to select where ``cond`` is ``False``.

        Returns:
            If ``x`` and ``y`` are ``None``, the indices where ``cond`` is ``True``. Otherwise, an array with
            elements taken from ``x`` where ``cond`` is ``True`` and from ``y`` elsewhere.

        """
        raise NotImplementedError

    @staticmethod
    def nonzero(array):
        """
        Args:
            array: an array.

        Returns:
            The indices of the nonzero elements of ``array``.

        """
        raise NotImplementedError

    @staticmethod
    def abs(array):
        """
        Args:
            array: an array.

        Returns:
            The element-wise absolute value of ``array``.

        """
        raise NotImplementedError

    @staticmethod
    def exp(array):
        """
        Args:
            array: an array.

        Returns:
            The element-wise exponential of ``array``.

        """
        raise NotImplementedError

    @staticmethod
    def sqrt(array):
        """
        Args:
            array: an array.

        Returns:
            The element-wise square root of ``array``.

        """
        raise NotImplementedError

    @staticmethod
    def clip(array, min, max):
        """
        Args:
            array: an array;
            min: lower bound;
            max: upper bound.

        Returns:
            ``array`` with values clipped to the ``[min, max]`` range.

        """
        raise NotImplementedError

    @staticmethod
    def sum(array, dim=None):
        """
        Args:
            array: an array;
            dim (None): dimension along which the sum is computed. If ``None``, the sum over the whole array
                is returned.

        Returns:
            The sum of ``array`` along ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def median(array):
        """
        Args:
            array: an array.

        Returns:
            The median of the elements of ``array``.

        """
        raise NotImplementedError

    @staticmethod
    def max(array, dim=None):
        """
        Args:
            array: an array;
            dim (None): dimension along which the maximum is computed. If ``None``, the maximum over the
                whole array is returned.

        Returns:
            The maximum value(s) of ``array`` along ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def min(array, dim=None):
        """
        Args:
            array: an array;
            dim (None): dimension along which the minimum is computed. If ``None``, the minimum over the
                whole array is returned.

        Returns:
            The minimum value(s) of ``array`` along ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def norm(array, ord=None, dim=None):
        """
        Args:
            array: an array;
            ord (None): order of the norm (see ``numpy.linalg.norm``/``torch.linalg.norm`` for the accepted
                values). If ``None``, the default order for the underlying library is used;
            dim (None): dimension along which the norm is computed. If ``None``, the norm of the flattened
                array is returned.

        Returns:
            The norm of ``array`` along ``dim``.

        """
        raise NotImplementedError

    @staticmethod
    def maximum(x, y):
        """
        Args:
            x: an array;
            y: an array.

        Returns:
            The element-wise maximum of ``x`` and ``y``.

        """
        raise NotImplementedError

    @staticmethod
    def minimum(x, y):
        """
        Args:
            x: an array;
            y: an array.

        Returns:
            The element-wise minimum of ``x`` and ``y``.

        """
        raise NotImplementedError

    @staticmethod
    def logical_and(x, y):
        """
        Args:
            x: a boolean array;
            y: a boolean array.

        Returns:
            The element-wise logical AND of ``x`` and ``y``.

        """
        raise NotImplementedError

    @staticmethod
    def rand(*dims, device=None):
        """
        Args:
            *dims: shape of the array to create;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            An array of shape ``dims`` sampled uniformly in ``[0, 1)``.

        """
        raise NotImplementedError

    @staticmethod
    def randint(low, high, size, device=None):
        """
        Args:
            low: lowest (inclusive) integer to be drawn;
            high: highest (exclusive) integer to be drawn;
            size: shape of the array to create;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            An array of shape ``size`` with random integers in ``[low, high)``.

        """
        raise NotImplementedError

    @staticmethod
    def multinomial(p):
        """
        Args:
            p: a 1D array of (unnormalized) probabilities.

        Returns:
            A single index sampled according to the probabilities in ``p``.

        """
        raise NotImplementedError

    @staticmethod
    def uniform(low, high):
        """
        Args:
            low: lower bound(s) of the uniform distribution;
            high: upper bound(s) of the uniform distribution.

        Returns:
            An array sampled uniformly between ``low`` and ``high``.

        """
        raise NotImplementedError

    @staticmethod
    def arange(start, stop, step=1, dtype=None, device=None):
        """
        Args:
            start: start of the interval;
            stop: end of the interval (exclusive);
            step (1): spacing between values;
            dtype (None): data type of the array;
            device (None): device the array should be allocated on. Only meaningful for :class:`TorchBackend`.

        Returns:
            An array with evenly spaced values within the given interval.

        """
        raise NotImplementedError

class NumpyBackend(ArrayBackend):
    """
    Array backend storing data in NumPy ``ndarray`` objects. It is the default backend for CPU-based
    environments and agents.

    """
    _DTYPE_MAP = {
        torch.bool:    np.dtype('bool'),
        torch.uint8:   np.dtype('uint8'),
        torch.int8:    np.dtype('int8'),
        torch.int16:   np.dtype('int16'),
        torch.int32:   np.dtype('int32'),
        torch.int64:   np.dtype('int64'),
        torch.float16: np.dtype('float16'),
        torch.float32: np.dtype('float32'),
        torch.float64: np.dtype('float64'),
    }

    @staticmethod
    def get_backend_name():
        return 'numpy'

    @staticmethod
    def get_backend_serialization():
        return 'numpy'

    @staticmethod
    def convert_to_backend(backend, array):
        return backend.to_numpy(array)

    @staticmethod
    def to_numpy(array):
        return array

    @staticmethod
    def to_torch(array):
        if array is None:
            return None
        torch_dtype = TorchBackend.to_backend_dtype(array.dtype)
        return torch.as_tensor(array, dtype=torch_dtype, device=TorchUtils.get_device())

    @staticmethod
    def to_list(array):
        return array.tolist()

    @staticmethod
    def as_array(array):
        return np.asarray(array)

    @staticmethod
    def from_list(array):
        return np.array(array)

    @classmethod
    def to_backend_dtype(cls, dtype):
        if isinstance(dtype, np.dtype):
            return dtype
        if isinstance(dtype, torch.dtype):
            return cls._DTYPE_MAP[dtype]
        return np.dtype(dtype)

    @staticmethod
    def empty(shape, device=None):
        return np.empty(shape)

    @staticmethod
    def full(shape, value):
        return np.full(shape, value)

    @staticmethod
    def concatenate(list_of_arrays, dim=0):
        return np.concatenate(list_of_arrays, axis=dim)

    @staticmethod
    def flatten(array):
        shape = array.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        return array.reshape(new_shape, order='F')

    @staticmethod
    def pack_padded_sequence(array, mask):
        shape = array.shape

        new_shape = (shape[0] * shape[1],) + shape[2:]
        return array.reshape(new_shape, order='F')[mask.flatten(order='F')]

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
    def masked_init(mask, values):
        result = np.empty((mask.shape[0],) + values.shape[1:])
        result[mask] = values
        return result

    @staticmethod
    def shape(array):
        return array.shape

    @staticmethod
    def size(arr):
        return np.size(arr)

    @staticmethod
    def none():
        return np.nan

    @staticmethod
    def inf():
        return np.inf

    @staticmethod
    def copy(array):
        return array.copy()

    @staticmethod
    def squeeze(array, dim=None):
        return np.squeeze(array, axis=dim)

    @staticmethod
    def expand_dims(array, dim):
        return np.expand_dims(array, axis=dim)

    @staticmethod
    def atleast_2d(array):
        return np.atleast_2d(array)

    @staticmethod
    def repeat(array, repeats):
        return np.repeat(array, repeats)

    @staticmethod
    def stack(lst, dim):
        return np.stack(lst, axis=dim)

    @staticmethod
    def where(cond, x=None, y=None):
        assert (x is None) == (y is None), "Either both or neither of x and y should be given."
        if x is None:
            return np.where(cond)
        else:
            return np.where(cond, x, y)

    @staticmethod
    def nonzero(array):
        return np.flatnonzero(array)

    @staticmethod
    def abs(array):
        return np.abs(array)

    @staticmethod
    def exp(array):
        return np.exp(array)

    @staticmethod
    def sqrt(array):
        return np.sqrt(array)

    @staticmethod
    def clip(array, min, max):
        return np.clip(array, min, max)

    @staticmethod
    def sum(array, dim=None):
        return np.sum(array, axis=dim)

    @staticmethod
    def median(array):
        return np.median(array)

    @staticmethod
    def max(array, dim=None):
        return np.max(array, axis=dim)

    @staticmethod
    def min(array, dim=None):
        return np.min(array, axis=dim)

    @staticmethod
    def norm(array, ord=None, dim=None):
        return np.linalg.norm(array, ord=ord, axis=dim)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)

    @staticmethod
    def minimum(x, y):
        return np.minimum(x, y)

    @staticmethod
    def logical_and(x, y):
        return np.logical_and(x, y)

    @classmethod
    def rand(cls, *dims, device=None):
        cls.check_device(device)
        return np.random.rand(*dims)

    @classmethod
    def randint(cls, low, high, size, device=None):
        assert type(size) == tuple
        cls.check_device(device)
        return np.random.randint(low, high, size)

    @staticmethod
    def multinomial(p):
        return np.array([np.random.choice(len(p), p=p)])

    @staticmethod
    def uniform(low, high):
        return np.random.uniform(low, high)

    @classmethod
    def arange(cls, start, stop, step=1, dtype=None, device=None):
        cls.check_device(device)
        return np.arange(start, stop, step, dtype=dtype)

class TorchBackend(ArrayBackend):
    """
    Array backend storing data in PyTorch ``Tensor`` objects. It supports GPU execution and is the backend of
    choice for neural-network based agents and vectorized environments running on device.

    """
    _DTYPE_MAP = {
        np.dtype('bool'):    torch.bool,
        np.dtype('uint8'):   torch.uint8,
        np.dtype('int8'):    torch.int8,
        np.dtype('int16'):   torch.int16,
        np.dtype('int32'):   torch.int32,
        np.dtype('int64'):   torch.int64,
        np.dtype('float16'): torch.float16,
        np.dtype('float32'): torch.float32,
        np.dtype('float64'): torch.float32,
    }

    @staticmethod
    def get_backend_name():
        return 'torch'

    @staticmethod
    def get_backend_serialization():
        return 'torch'

    @staticmethod
    def convert_to_backend(backend, array):
        return backend.to_torch(array)

    @staticmethod
    def to_numpy(array):
        return None if array is None else array.detach().cpu().numpy()

    @staticmethod
    def to_torch(array):
        return array

    @staticmethod
    def to_list(array):
        return array.tolist()

    @staticmethod
    def as_array(array):
        return torch.as_tensor(array, device=TorchUtils.get_device())

    @staticmethod
    def from_list(array):
        if len(array) > 0 and isinstance(array[0], torch.Tensor):
            return torch.stack(array)
        else:
            return torch.tensor(array).to(TorchUtils.get_device())

    @classmethod
    def to_backend_dtype(cls, dtype):
        if isinstance(dtype, torch.dtype):
            return dtype
        return cls._DTYPE_MAP[np.dtype(dtype)]

    @staticmethod
    def empty(shape, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.empty(shape, device=device)

    @staticmethod
    def full(shape, value):
        return torch.full(shape, value).to(device=TorchUtils.get_device())

    @staticmethod
    def concatenate(list_of_arrays, dim=0):
        return torch.concat(list_of_arrays, dim=dim)

    @staticmethod
    def flatten(array):
        shape = array.shape
        new_shape = (shape[0]*shape[1], ) + shape[2:]
        return array.transpose(0, 1).reshape(new_shape)

    @staticmethod
    def pack_padded_sequence(array, mask):
        shape = array.shape

        new_shape = (shape[0]*shape[1], ) + shape[2:]

        return array.transpose(0, 1).reshape(new_shape)[mask.transpose(0, 1).flatten()]

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
    def masked_init(mask, values):
        result = torch.empty((mask.shape[0],) + values.shape[1:], device=TorchUtils.get_device())
        result[mask] = torch.as_tensor(values, dtype=result.dtype, device=TorchUtils.get_device())
        return result

    @staticmethod
    def shape(array):
        return array.shape

    @staticmethod
    def size(arr):
        return torch.numel(arr)

    @staticmethod
    def none():
        return torch.nan

    @staticmethod
    def inf():
        return torch.inf

    @staticmethod
    def copy(array):
        return array.clone()

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
    def atleast_2d(array):
        return torch.atleast_2d(array)

    @staticmethod
    def repeat(array, repeats):
        return torch.repeat_interleave(array, repeats)

    @staticmethod
    def stack(lst, dim):
        return torch.stack(lst, dim=dim)

    @staticmethod
    def where(cond, x=None, y=None):
        assert (x is None) == (y is None), "Either both or neither of x and y should be given."
        if x is None:
            return torch.where(cond)
        else:
            return torch.where(cond, x, y)

    @staticmethod
    def nonzero(array):
        return torch.nonzero(array)

    @staticmethod
    def abs(array):
        return torch.abs(array)

    @staticmethod
    def exp(array):
        return torch.exp(array)

    @staticmethod
    def sqrt(array):
        return torch.sqrt(array)

    @staticmethod
    def clip(array, min, max):
        return torch.clip(array, min, max)

    @staticmethod
    def sum(array, dim=None):
        return torch.sum(array, dim=dim)

    @staticmethod
    def median(array):
        return array.median()

    @staticmethod
    def max(array, dim=None):
        return torch.max(array) if dim is None else torch.max(array, dim=dim).values

    @staticmethod
    def min(array, dim=None):
        return torch.min(array) if dim is None else torch.min(array, dim=dim).values

    @staticmethod
    def norm(array, ord=None, dim=None):
        return torch.linalg.norm(array, ord=ord, dim=dim)

    @staticmethod
    def maximum(x, y):
        return torch.maximum(x, y)

    @staticmethod
    def minimum(x, y):
        return torch.minimum(x, y)

    @staticmethod
    def logical_and(x, y):
        return torch.logical_and(x, y)

    @staticmethod
    def rand(*dims, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.rand(dims, device=device)

    @staticmethod
    def randint(low, high, size, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.randint(low, high, size, device=device)

    @staticmethod
    def multinomial(p):
        return torch.multinomial(p, 1)

    @staticmethod
    def uniform(low, high):
        low = torch.as_tensor(low, device=TorchUtils.get_device())
        high = torch.as_tensor(high, device=TorchUtils.get_device())
        return low + (high - low) * torch.rand_like(low)

    @classmethod
    def arange(cls, start, stop, step=1, dtype=None, device=None):
        device = TorchUtils.get_device() if device is None else device
        return torch.arange(start, stop, step, dtype=dtype, device=device)

class ListBackend(ArrayBackend):
    """
    Storage backend that keeps data in plain Python lists. It grows without pre-allocation (which allows
    collecting episodes of unbounded/infinite horizon) and can hold ragged or non-array observations and
    actions. It is numpy-serialized: container reshaping (``empty``, ``full``, ``concatenate``, ``flatten``,
    ``pack_padded_sequence``) is list-native and ragged-safe, while any numeric computation materializes the
    (regular) data to numpy via ``as_array``.

    """
    @staticmethod
    def get_backend_name():
        return 'list'

    @staticmethod
    def get_backend_serialization():
        return 'numpy'

    @staticmethod
    def convert_to_backend(backend, array):
        return array

    @staticmethod
    def to_numpy(array):
        return np.array(array)

    @staticmethod
    def to_torch(array):
        return None if array is None else torch.as_tensor(array, device=TorchUtils.get_device())

    @staticmethod
    def to_list(array):
        return array

    @staticmethod
    def as_array(array):
        return np.array(array)

    @staticmethod
    def from_list(array):
        return array

    @staticmethod
    def to_backend_dtype(dtype):
        return dtype

    @staticmethod
    def empty(shape, device=None):
        if len(shape) == 0:
            return None
        return [ListBackend.empty(shape[1:]) for _ in range(shape[0])]

    @staticmethod
    def full(shape, value):
        if len(shape) == 0:
            return value
        return [ListBackend.full(shape[1:], value) for _ in range(shape[0])]

    @staticmethod
    def concatenate(list_of_arrays, dim=0):
        result = []
        for array in list_of_arrays:
            result += list(array)
        return result

    @staticmethod
    def flatten(array):
        n_steps = len(array)
        n_envs = len(array[0])
        return [array[s][e] for e in range(n_envs) for s in range(n_steps)]

    @staticmethod
    def pack_padded_sequence(array, mask):
        n_steps = len(array)
        n_envs = mask.shape[1]
        return [array[s][e] for e in range(n_envs) for s in range(n_steps) if mask[s, e]]

    @classmethod
    def zeros(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return NumpyBackend.zeros(*dims, dtype=dtype)

    @classmethod
    def ones(cls, *dims, dtype=float, device=None):
        cls.check_device(device)
        return NumpyBackend.ones(*dims, dtype=dtype)

    @classmethod
    def zeros_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return NumpyBackend.zeros_like(array, dtype=dtype)

    @classmethod
    def ones_like(cls, array, dtype=float, device=None):
        cls.check_device(device)
        return NumpyBackend.ones_like(array, dtype=dtype)

    @staticmethod
    def masked_init(mask, values):
        result = [None] * len(mask)
        for j, i in enumerate(NumpyBackend.nonzero(mask)):
            result[int(i)] = values[j]
        return result

    @staticmethod
    def shape(array):
        return NumpyBackend.shape(NumpyBackend.as_array(array))

    @staticmethod
    def size(arr):
        return NumpyBackend.size(arr)

    @staticmethod
    def none():
        return None

    @staticmethod
    def inf():
        return NumpyBackend.inf()

    @staticmethod
    def copy(array):
        return copy.deepcopy(array)

    @staticmethod
    def squeeze(array, dim=None):
        return NumpyBackend.squeeze(array, dim)

    @staticmethod
    def expand_dims(array, dim):
        return NumpyBackend.expand_dims(array, dim)

    @staticmethod
    def atleast_2d(array):
        return NumpyBackend.atleast_2d(array)

    @staticmethod
    def repeat(array, repeats):
        return NumpyBackend.repeat(array, repeats)

    @staticmethod
    def stack(lst, dim):
        return NumpyBackend.stack(lst, dim)

    @staticmethod
    def where(cond, x=None, y=None):
        return NumpyBackend.where(cond, x, y)

    @staticmethod
    def nonzero(array):
        return NumpyBackend.nonzero(array)

    @staticmethod
    def abs(array):
        return NumpyBackend.abs(array)

    @staticmethod
    def exp(array):
        return NumpyBackend.exp(array)

    @staticmethod
    def sqrt(array):
        return NumpyBackend.sqrt(array)

    @staticmethod
    def clip(array, min, max):
        return NumpyBackend.clip(array, min, max)

    @staticmethod
    def sum(array, dim=None):
        return NumpyBackend.sum(array, dim)

    @staticmethod
    def median(array):
        return NumpyBackend.median(array)

    @staticmethod
    def max(array, dim=None):
        return NumpyBackend.max(array, dim)

    @staticmethod
    def min(array, dim=None):
        return NumpyBackend.min(array, dim)

    @staticmethod
    def norm(array, ord=None, dim=None):
        return NumpyBackend.norm(array, ord=ord, dim=dim)

    @staticmethod
    def maximum(x, y):
        return NumpyBackend.maximum(x, y)

    @staticmethod
    def minimum(x, y):
        return NumpyBackend.minimum(x, y)

    @staticmethod
    def logical_and(x, y):
        return NumpyBackend.logical_and(x, y)

    @classmethod
    def rand(cls, *dims, device=None):
        cls.check_device(device)
        return NumpyBackend.rand(*dims)

    @classmethod
    def randint(cls, low, high, size, device=None):
        cls.check_device(device)
        return NumpyBackend.randint(low, high, size)

    @staticmethod
    def multinomial(p):
        return NumpyBackend.multinomial(p)

    @staticmethod
    def uniform(low, high):
        return NumpyBackend.uniform(low, high)

    @classmethod
    def arange(cls, start, stop, step=1, dtype=None, device=None):
        cls.check_device(device)
        return NumpyBackend.arange(start, stop, step, dtype=dtype)
