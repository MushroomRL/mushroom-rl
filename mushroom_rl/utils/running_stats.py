import numpy as np
from collections import deque


class RunningStandardization:
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

    def get_state(self):
        """
        Returns:
            A dictionary containing the state of the filter.

        """
        return dict(mean=self._m, var=self._s, count=self._n)

    def set_state(self, state):
        """
        Set the state of the filter.

        """
        self._m = state["mean"]
        self._s = state["var"]
        self._n = state["count"]

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


class RunningExpWeightedAverage:
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


class RunningAveragedWindow:
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
        self.shape = shape
        self.window_size = window_size
        self.reset(init_value)

    def reset(self, init_value=None):
        """
        Reset the window.

        Args:
            init_value (np.ndarray): initial value of the filter.

        """
        if init_value is None:
            self.avg_buffer = deque(np.zeros((1, *self.shape)),
                                    maxlen=self.window_size)
        else:
            self.avg_buffer = deque([init_value], maxlen=self.window_size)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        self.avg_buffer.append(value)

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return np.mean(self.avg_buffer, axis=0)
