from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class AbstractTiles(MushroomObject):
    """
    Base class for the tiles. A tiling maps a point of the state space to the index of the corresponding tile.

    """
    def __init__(self, state_components=None):
        """
        Constructor.

        Args:
            state_components (list, None): list of the dimensions of the input to be considered by the tiling. If
                None, every dimension of the input is used.

        """
        self._state_components = state_components

        self._add_save_attr(_state_components='primitive')

    def __call__(self, x):
        """
        Compute the index of the tile corresponding to each input point, considering only the dimensions of the
        input selected by ``state_components``.

        Args:
            x (np.ndarray): input batch with shape ``(n_samples, n_dimensions)``.

        Returns:
            The index of the corresponding tile for each sample, with shape ``(n_samples,)``, or ``-1`` for the
            samples falling outside the tiling.

        """
        return self._compute(self._select(x))

    @property
    def size(self):
        """
        Returns:
            The number of tiles of the tiling.

        """
        raise NotImplementedError

    def _compute(self, x):
        """
        Compute the index of the tile corresponding to each input point, already restricted to the selected
        dimensions.

        Args:
            x (np.ndarray): input batch with shape ``(n_samples, len(state_components))``.

        Returns:
            The index of the corresponding tile for each sample, with shape ``(n_samples,)``, or ``-1`` for the
            samples falling outside the tiling.

        """
        raise NotImplementedError

    def _select(self, x):
        if self._state_components is not None:
            return x[:, self._state_components]

        return x

    @staticmethod
    def _to_numpy(*arrays):
        """
        Convert the given arrays, e.g. the prototypes of the tiling, into the numpy arrays the tiles are built
        from, whatever the backend they are provided in.

        Args:
            *arrays: one or more arrays to convert.

        Returns:
            The converted array, or a tuple of converted arrays if more than one was passed.

        """
        return ArrayBackend.convert(*arrays, to='numpy')
