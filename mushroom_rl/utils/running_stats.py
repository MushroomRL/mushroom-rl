import numpy as np
from collections import deque
from mushroom_rl.core import Serializable


class RunningStandardization(Serializable):
    """
    Compute a running standardization of values according to Welford's online
    algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, alpha=1e-32):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float, 1e-32): minimum learning rate.

        """
        self._shape = shape

        assert 0. < alpha < 1.
        self._alpha = alpha

        self._n = 1
        self._m = np.zeros(self._shape)
        self._s = np.ones(self._shape)

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _n='primitive',
            _m='primitive',
            _s='primitive',
        )

    def reset(self):
        """
        Reset the mean and standard deviation.

        """
        self._n = 1
        self._m = np.zeros(self._shape)
        self._s = np.ones(self._shape)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        self._n += 1
        alpha = max(1. / self._n, self._alpha)
        new_m = (1 - alpha) * self._m + alpha * value
        new_s = self._s + (value - self._m) * (value - new_m)
        self._m, self._s = new_m, new_s

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._m

    @property
    def std(self):
        """
        Returns:
            The estimated standard deviation value.

        """
        return np.sqrt(self._s / self._n)


class RunningExpWeightedAverage(Serializable):
    """
    Compute an exponentially weighted moving average.

    """
    def __init__(self, shape, alpha, init_value=None):
        """
        Constructor.

        Args:
             shape (tuple): shape of the data to standardize;
             alpha (float): learning rate;
             init_value (np.ndarray): initial value of the filter.

        """
        self._shape = shape
        self._alpha = alpha
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _avg_value='primitive',
        )

    def reset(self, init_value=None):
        """
        Reset the mean and standard deviation.

        Args:
            init_value (np.ndarray): initial value of the filter.

        """
        if init_value is None:
            self._avg_value = np.zeros(self._shape)
        else:
            self._avg_value = init_value

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        self._avg_value = (
            1. - self._alpha) * self._avg_value + self._alpha * value

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
    def __init__(self, shape, window_size, init_value=None):
        """
        Constructor.

        Args:
             shape (tuple): shape of the data to standardize;
             window_size (int): size of the windos;
             init_value (np.ndarray): initial value of the filter.

        """
        self._shape = shape
        self._window_size = window_size
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _window_size='primitive',
            _avg_buffer='primitive',
        )

    def reset(self, init_value=None):
        """
        Reset the window.

        Args:
            init_value (np.ndarray): initial value of the filter.

        """
        if init_value is None:
            self._avg_buffer = deque(np.zeros((1, *self._shape)),
                                    maxlen=self._window_size)
        else:
            self._avg_buffer = deque([init_value], maxlen=self._window_size)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        self._avg_buffer.append(value)

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return np.mean(self._avg_buffer, axis=0)
