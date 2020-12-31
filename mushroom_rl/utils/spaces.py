import numpy as np

from mushroom_rl.core import Serializable


class Box(Serializable):
    """
    This class implements functions to manage continuous states and action
    spaces. It is similar to the ``Box`` class in ``gym.spaces.box``.

    """
    def __init__(self, low, high, shape=None):
        """
        Constructor.

        Args:
            low ([float, np.ndarray]): the minimum value of each dimension of
                the space. If a scalar value is provided, this value is
                considered as the minimum one for each dimension. If a
                np.ndarray is provided, each i-th element is considered the
                minimum value of the i-th dimension;
            high ([float, np.ndarray]): the maximum value of dimensions of the
                space. If a scalar value is provided, this value is considered
                as the maximum one for each dimension. If a np.ndarray is
                provided, each i-th element is considered the maximum value
                of the i-th dimension;
            shape (np.ndarray, None): the dimension of the space. Must match
                the shape of ``low`` and ``high``, if they are np.ndarray.

        """
        if shape is None:
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = low
            self._high = high
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

        self._add_save_attr(
            _low='numpy',
            _high='numpy'
        )

    @property
    def low(self):
        """
        Returns:
             The minimum value of each dimension of the space.

        """
        return self._low

    @property
    def high(self):
        """
        Returns:
             The maximum value of each dimension of the space.

        """
        return self._high

    @property
    def shape(self):
        """
        Returns:
            The dimensions of the space.

        """
        return self._shape

    def _post_load(self):
        self._shape = self._low.shape


class Discrete(Serializable):
    """
    This class implements functions to manage discrete states and action
    spaces. It is similar to the ``Discrete`` class in ``gym.spaces.discrete``.

    """
    def __init__(self, n):
        """
        Constructor.

        Args:
            n (int): the number of values of the space.

        """
        self.values = np.arange(n)
        self.n = n

        self._add_save_attr(
            n='primitive',
            values='numpy'
        )

    @property
    def size(self):
        """
        Returns:
            The number of elements of the space.

        """
        return self.n,

    @property
    def shape(self):
        """
        Returns:
            The shape of the space that is always (1,).

        """
        return 1,
