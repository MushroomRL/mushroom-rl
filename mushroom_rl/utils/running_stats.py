import numpy as np
from collections import deque


class RunningStandardization:
    """
    Computes a running standardization of vales according to Welford's
        online algorithm:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, alpha=1e-32):
        self.shape = shape

        assert 0.0 < alpha < 1.0
        self.alpha = alpha

        self.n = 1
        self._m = np.zeros(self.shape)
        self._s = np.ones(self.shape)

    def reset(self):
        self.n = 1
        self._m = np.zeros(self.shape)
        self._s = np.ones(self.shape)

    def update_stats(self, values):
        self.n += 1
        alpha = max(1.0 / self.n, self.alpha)
        new_m = (1 - alpha) * self._m + alpha * values
        new_s = self._s + (values - self._m) * (values - new_m)
        self._m, self._s = new_m, new_s

    @property
    def mean(self):
        return self._m

    @property
    def std(self):
        return np.sqrt(self._s / self.n)

    def get_state(self):
        return dict(mean=self._m, std=self._s, count=self.n)

    def set_state(self, state):
        self._m = state["mean"]
        self._s = state["std"]
        self.n = state["count"]


class RunningExpWeightedWindow:
    """
    Computes a Exponentially Weighted Moving Average.

    """
    def __init__(self, shape, alpha, init_value=None):
        self.shape = shape
        self.alpha = alpha
        self.reset(init_value)

    def reset(self, init_values=None):
        if init_values is None:
            self.avg_value = np.zeros(self.shape)
        else:
            self.avg_value = init_values

    def update_stats(self, values):
        self.avg_value = (1.0 - self.alpha) * self.avg_value + self.alpha * values
        return self.avg_value

    def get_avg_value(self):
        return self.avg_value


class RunningAveragedWindow:
    """
    Computes a Simple Moving Average.

    """
    def __init__(self, shape, window_size, init_values=None):
        self.shape = shape
        self.window_size = window_size
        self.reset(init_values)

    def reset(self, init_values=None):
        if init_values is None:
            self.avg_buffer = deque(np.zeros((1, *self.shape)), maxlen=self.window_size)
        else:
            self.avg_buffer = deque([init_values], maxlen=self.window_size)

    def update_stats(self, values):
        self.avg_buffer.append(values)

    def get_avg_value(self):
        return np.mean(self.avg_buffer, axis=0)