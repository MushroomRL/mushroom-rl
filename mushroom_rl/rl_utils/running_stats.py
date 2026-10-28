from collections import deque
from mushroom_rl.core import MushroomBackendObject


class RunningStandardization(MushroomBackendObject):
    """
    Compute a running standardization of values according to Welford's online algorithm.

    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, alpha=1e-32, backend='numpy', device=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float, 1e-32): minimum learning rate;
            backend (str, 'numpy'): name of the backend to be used;
            device (str, None): device the statistics are stored on.

        """
        self._s = None
        self._m = None
        self._n = None

        self._shape = shape

        assert 0. < alpha < 1.
        self._alpha = alpha

        super().__init__(backend, device)

        self.reset()

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _n='primitive',
            _m=self._backend.get_backend_serialization(),
            _s=self._backend.get_backend_serialization()
        )

    def reset(self):
        """
        Reset the mean and standard deviation.

        """
        self._n = 1
        self._m = self._backend.zeros(*self._shape, device=self._device)
        self._s = self._backend.ones(*self._shape, device=self._device)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (Array): current data value to use for the update.

        """
        value = self._backend.atleast_2d(value)
        batch_size = len(value)
        batch_mean = value.mean(0)
        batch_m2 = ((value - batch_mean) ** 2).sum(0)
        n_old = self._n
        self._n += batch_size
        alpha = max(batch_size / self._n, self._alpha)
        new_m = (1 - alpha) * self._m + alpha * batch_mean
        new_s = self._s + batch_m2 + (batch_mean - self._m) ** 2 * n_old * batch_size / self._n
        self._m, self._s = new_m, new_s

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._backend.squeeze(self._m)

    @property
    def std(self):
        """
        Returns:
            The estimated standard deviation value.

        """
        return self._backend.squeeze(self._backend.sqrt(self._s / self._n))


class RunningExpWeightedAverage(MushroomBackendObject):
    """
    Compute an exponentially weighted moving average.

    """
    def __init__(self, shape, alpha, backend, device=None, init_value=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float): learning rate;
            backend (str): name of the backend to be used;
            device (str, None): device the statistics are stored on;
            init_value (np.ndarray): initial value of the filter.

        """
        self._avg_value = None

        self._shape = shape
        self._alpha = alpha
        super().__init__(backend, device)
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _alpha='primitive',
            _avg_value=self._backend.get_backend_serialization(),
        )

    def reset(self, init_value=None):
        """
        Reset the mean and standard deviation.

        Args:
            init_value (Array): initial value of the filter.

        """
        if init_value is None:
            self._avg_value = self._backend.zeros(1, *self._shape, device=self._device)
        else:
            init_value = self._backend.convert(init_value, device=self._device)
            self._avg_value = self._backend.atleast_2d(init_value)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (Array): current data value to use for the update.

        """
        value = self._backend.atleast_2d(value)
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


class RunningAveragedWindow(MushroomBackendObject):
    """
    Compute the running average using a window of fixed size.

    """
    def __init__(self, shape, window_size, backend, device=None, init_value=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            window_size (int): size of the windows;
            backend (str): name of the backend to be used;
            device (str, None): device the statistics are stored on;
            init_value (np.ndarray): initial value of the filter.

        """
        self._avg_buffer = None

        self._shape = shape
        self._window_size = window_size
        super().__init__(backend, device)
        self.reset(init_value)

        self._add_save_attr(
            _shape='primitive',
            _window_size='primitive',
            _avg_buffer='torch' if self._backend.get_backend_serialization() == 'torch' else 'primitive',
        )

    def reset(self, init_value=None):
        """
        Reset the window.

        Args:
            init_value (np.ndarray): initial value of the filter.

        """
        if init_value is None:
            self._avg_buffer = deque(self._backend.zeros(1, *self._shape, device=self._device),
                                     maxlen=self._window_size)
        else:
            self._avg_buffer = deque([self._backend.convert(init_value, device=self._device)],
                                     maxlen=self._window_size)

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (np.ndarray): current data value to use for the update.

        """
        value = self._backend.atleast_2d(value)
        batch_size = len(value)
        for i in range(batch_size):
            self._avg_buffer.append(value[i])

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._backend.convert(self._avg_buffer, device=self._device).mean(0)
