import copy
from collections import deque

import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend


class MushroomBackendObject(MushroomObject):
    """
    Base class of the MushroomRL objects storing their data in an array backend. The device is never saved: a loaded
    object is placed on the device requested at load time, or on the default one.

    """
    def __init__(self, backend, device=None):
        """
        Constructor.

        Args:
            backend (str): name of the array backend the object data is stored in;
            device (str, None): device the object data is stored on, or ``None`` for the default one. Only allowed
                with the torch backend.

        Raises:
            ValueError: if ``device`` is not ``None`` and the backend does not support devices.

        """
        self._backend = ArrayBackend.get_array_backend(backend)
        self._device = self._backend.check_device(device)

        self._add_save_attr(
            _backend='primitive',
            _device='none'
        )

    def to(self, device):
        """
        Move the object to another device, together with the objects it owns, and rebuild the state that is rebuilt
        after a load. Every runtime state is wiped. The objects it only references, owned by another
        object, are not moved.

        Args:
            device (str, None): the device to move to, or ``None`` for the default one.

        Returns:
            This object.

        """
        self._restore_device(device)

        for name, method in self._save_attributes.items():
            if not method.endswith('!'):
                value = getattr(self, name, None)
                if method == 'torch' and self._device is not None:
                    setattr(self, name, self._move_to_device(value, self._device))
                elif method == 'mushroom':
                    for item in value if isinstance(value, list) else [value]:
                        if hasattr(item, 'to'):
                            item.to(device)
                elif method == 'none' and name != '_device':
                    setattr(self, name, None)

        self._post_load(device)

        return self

    @property
    def backend(self):
        """
        Returns:
            The array backend the object data is stored in.

        """
        return self._backend

    @property
    def backend_name(self):
        """
        Returns:
            The name of the array backend the object data is stored in.

        """
        return self._backend.get_backend_name()

    @property
    def device(self):
        """
        Returns:
            The device the object data is stored on, or ``None`` for the backends that do not support devices.

        """
        return self._device

    def _restore_device(self, device):
        self._device = self._backend.resolve_device(device)

    @staticmethod
    def _move_to_device(value, device):
        if isinstance(value, torch.nn.Parameter):
            value.data = value.data.to(device)
            if value.grad is not None:
                value.grad = value.grad.to(device)
            return value
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            return value.to(device)
        if isinstance(value, (dict, list, deque)):
            moved = copy.copy(value)
            keys = list(moved.keys()) if isinstance(moved, dict) else range(len(moved))
            for key in keys:
                moved[key] = MushroomBackendObject._move_to_device(moved[key], device)
            return moved
        if isinstance(value, tuple):
            items = [MushroomBackendObject._move_to_device(item, device) for item in value]
            return value._make(items) if hasattr(value, '_make') else tuple(items)

        return value
