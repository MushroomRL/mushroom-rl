from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class BasisFunction(MushroomObject):
    """
    Base class for the numpy basis functions. A basis function maps a batch of raw inputs to a single feature of the
    feature vector.

    """
    def __init__(self, dimensions=None):
        """
        Constructor.

        Args:
            dimensions (list, None): list of the dimensions of the input to be considered by the feature.

        """
        self._dim = dimensions

        self._add_save_attr(_dim='primitive')

    def __call__(self, x):
        """
        Evaluate the basis function on a batch of inputs, considering only the dimensions of the input selected by
        ``dimensions``.

        Args:
            x (np.ndarray): input batch with shape ``(n_samples, n_dimensions)``.

        Returns:
            The value of the basis function for each sample, with shape ``(n_samples,)``.

        """
        return self._compute(self._select(x))

    def _compute(self, x):
        """
        Evaluate the basis function on a batch of inputs, already restricted to the selected dimensions.

        Args:
            x (np.ndarray): input batch with shape ``(n_samples, len(dimensions))``.

        Returns:
            The value of the basis function for each sample, with shape ``(n_samples,)``.

        """
        raise NotImplementedError

    def _select(self, x):
        if self._dim is not None:
            return x[:, self._dim]
        return x

    @staticmethod
    def _to_numpy(*arrays):
        """
        Convert the given arrays, e.g. the bounds of the input space, into the numpy arrays the basis functions
        are built from, whatever the backend they are provided in.

        Args:
            *arrays: one or more arrays to convert.

        Returns:
            The converted array, or a tuple of converted arrays if more than one was passed.

        """
        return ArrayBackend.convert(*arrays, to='numpy')
