from collections import deque
from mushroom_rl.core import MushroomBackendObject


class RunningStatistics(MushroomBackendObject):
    """
    Base class for the online statistics of a signal, updated with single samples or with batches of them.

    """
    def __init__(self, shape, backend, device=None):
        """
        Constructor.

        Args:
            shape (tuple): shape of a single sample;
            backend (str): name of the backend to be used;
            device (str, None): device the statistics are stored on.

        """
        self._shape = shape

        super().__init__(backend, device)

        self._add_save_attr(_shape='primitive')

    def update_stats(self, value):
        """
        Update the statistics with the current data value.

        Args:
            value (Array): a single sample, or a batch of samples along the first axis.

        """
        self._update(self._as_batch(value))

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value, with the shape of a single sample.

        """
        raise NotImplementedError

    def _update(self, batch):
        """
        Update the statistics with a batch of samples.

        Args:
            batch (Array): the samples, along the first axis.

        """
        raise NotImplementedError

    def _as_batch(self, value):
        """
        Args:
            value (Array): a single sample, or a batch of samples along the first axis.

        Returns:
            ``value`` as a batch of samples along the first axis.

        """
        value = self._backend.atleast_2d(value)
        if value.ndim == len(self._shape):
            value = self._backend.expand_dims(value, 0)
        return value


class RunningStandardization(RunningStatistics):
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

        assert 0. < alpha < 1.
        self._alpha = alpha

        super().__init__(shape, backend, device)

        self.reset()

        self._add_save_attr(
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

    def _update(self, batch):
        batch_size = len(batch)
        batch_mean = batch.mean(0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(0)
        n_old = self._n
        self._n += batch_size
        alpha = max(batch_size / self._n, self._alpha)
        new_m = (1 - alpha) * self._m + alpha * batch_mean
        new_s = self._s + batch_m2 + (batch_mean - self._m) ** 2 * n_old * batch_size / self._n
        self._m, self._s = new_m, new_s


class RunningExpWeightedAverage(RunningStatistics):
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

        self._alpha = alpha
        super().__init__(shape, backend, device)
        self.reset(init_value)

        self._add_save_attr(
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
            self._avg_value = self._backend.zeros(*self._shape, device=self._device)
        else:
            self._avg_value = self._backend.convert(init_value, device=self._device)

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._avg_value

    def _update(self, batch):
        for i in range(len(batch)):
            self._avg_value = (1. - self._alpha) * self._avg_value + self._alpha * batch[i]


class RunningAveragedWindow(RunningStatistics):
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

        self._window_size = window_size
        super().__init__(shape, backend, device)
        self.reset(init_value)

        self._add_save_attr(
            _window_size='primitive',
            _avg_buffer=self._backend.get_container_serialization(),
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

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.

        """
        return self._backend.stack(list(self._avg_buffer), dim=0).mean(0)

    def _update(self, batch):
        for i in range(len(batch)):
            self._avg_buffer.append(batch[i])
