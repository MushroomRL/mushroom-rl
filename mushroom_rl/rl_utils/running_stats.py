import numpy as np
from collections import deque
from mushroom_rl.core import Serializable, ArrayBackend


class RunningStandardization(Serializable):
    """
    Compute a running standardization of values according to Welford's online
    algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, backend, alpha=1e-32):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            backend (str): name of the backend to be used;
            alpha (float, 1e-32): minimum learning rate.

        """
        assert backend in ["numpy", "torch"]
        self._shape = shape

        assert 0. < alpha < 1.
        self._alpha = alpha

        self._array_backend = ArrayBackend.get_array_backend(backend)

        self._n = 1
        self._m = self._array_backend.zeros(*self._shape)
        self._s = self._array_backend.ones(*self._shape)

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _array_backend='pickle',
            _n='primitive',
            _m='primitive',
            _s='primitive'
        )

    def reset(self):
        """
        Reset the mean and standard deviation.

        """
        self._n = 1
        self._m = self._array_backend.zeros(1, *self._shape)
        self._s = self._array_backend.ones(1, *self._shape)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (Array): current data value to use for the update.

        """
        value = self._array_backend.atleast_2d(value)
        batch_size = len(value)
        self._n += batch_size
        alpha = max(batch_size / self._n, self._alpha)
        new_m = (1 - alpha) * self._m + alpha * value.mean(0)
        new_s = self._s + (value.mean(0) - self._m) * (value.mean(0) - new_m)
        self._m, self._s = new_m, new_s

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._array_backend.squeeze(self._m)

    @property
    def std(self):
        """
        Returns:
            The estimated standard deviation value.

        """
        return self._array_backend.squeeze(self._array_backend.sqrt(self._s / self._n))


class RunningExpWeightedAverage(Serializable):
    """
    Compute an exponentially weighted moving average.

    """
    def __init__(self, shape, alpha, backend, init_value=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float): learning rate;
            backend (str): name of the backend to be used;
            init_value (np.ndarray): initial value of the filter.

        """
        assert backend in ["numpy", "torch"]
        self._shape = shape
        self._alpha = alpha
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _array_backend="pickle",
            _avg_value='primitive',
        )

    def reset(self, init_value=None):
        """
        Reset the mean and standard deviation.

        Args:
            init_value (Array): initial value of the filter.

        """
        if init_value is None:
            self._avg_value = self._array_backend.zeros(1, *self._shape)
        else:
            self._avg_value = self._array_backend.atleast_2d(self._array_backend.convert(init_value))

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (Array): current data value to use for the update.

        """
        value = self._array_backend.atleast_2d(value)
        batch_size = len(value)
        for i in range(batch_size):
            self._avg_value = (1. - self._alpha) * self._avg_value + self._alpha * value[i]

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._avg_value


class RunningAveragedWindow(Serializable):
    """
    Compute the running average using a window of fixed size.

    """
    def __init__(self, shape, window_size, backend, init_value=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            window_size (int): size of the windows;
            backend (str): name of the backend to be used;
            init_value (np.ndarray): initial value of the filter.

        """
        assert backend in ["numpy", "torch"]
        self._shape = shape
        self._window_size = window_size
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _window_size='primitive',
            _array_backend='pickle',
            _avg_buffer='primitive',
        )

    def reset(self, init_value=None):
        """
        Reset the window.

        Args:
            init_value (np.ndarray): initial value of the filter.

        """
        if init_value is None:
            self._avg_buffer = deque(self._array_backend.zeros(1, *self._shape),
                                     maxlen=self._window_size)
        else:
            self._avg_buffer = deque([self._array_backend.convert(init_value)], maxlen=self._window_size)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        value = self._array_backend.atleast_2d(value)
        batch_size = len(value)
        for i in range(batch_size):
            self._avg_buffer.append(value[i])

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._array_backend.convert(self._avg_buffer).mean(0)
