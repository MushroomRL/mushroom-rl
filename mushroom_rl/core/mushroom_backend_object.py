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
        self._device = self._backend.check_device(device if self._backend.supports_devices() else None)
