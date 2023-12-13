import numpy as np

from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.approximators.table import Table


class VarianceParameter(Parameter):
    """
    Abstract class to implement variance-dependent parameters. A ``target``
    parameter is expected.

    """
    def __init__(self, value, exponential=False, min_value=None, tol=1.,
                 size=(1,)):
        """
        Constructor.

        Args:
            tol (float): value of the variance of the target variable such that
                The parameter value is 0.5.

        """
        self._exponential = exponential
        self._tol = tol
        self._weights_var = Table(size)
        self._x = Table(size)
        self._x2 = Table(size)
        self._parameter_value = Table(size)

        super().__init__(value, min_value, size)

        self._add_save_attr(
            _exponential='primitive',
            _tol='primitive',
            _weights_var='mushroom',
            _x='mushroom',
            _x2='mushroom',
            _parameter_value='mushroom',
        )

    def _compute(self, *idx, **kwargs):
        return self._parameter_value[idx]

    def update(self, *idx, **kwargs):
        """
        Updates the value of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter whose number of visits has to be
                updated.
            target (float): Value of the target variable;
            factor (float): Multiplicative factor for the parameter value, useful
                when the parameter depend on another parameter value.

        """
        x = kwargs['target']
        factor = kwargs.get('factor', 1.)

        # compute parameter value
        n = self._n_updates[idx]
        self._n_updates[idx] += 1

        if n < 2:
            parameter_value = self._initial_value
        else:
            var = n * (self._x2[idx] - self._x[idx] ** 2) / (n - 1.)
            var_estimator = var * self._weights_var[idx]
            parameter_value = self._compute_parameter(var_estimator,
                                                      sigma_process=var,
                                                      index=idx)

        # update state
        self._x[idx] += (x - self._x[idx]) / self._n_updates[idx]
        self._x2[idx] += (x ** 2 - self._x2[idx]) / self._n_updates[idx]
        self._weights_var[idx] = (
            1. - factor * parameter_value) ** 2 * self._weights_var[idx] + (
            factor * parameter_value) ** 2
        self._parameter_value[idx] = parameter_value

    def _compute_parameter(self, sigma, **kwargs):
        raise NotImplementedError('VarianceParameter is an abstract class.')


class VarianceIncreasingParameter(VarianceParameter):
    """
    Class implementing a parameter that increases with the target
    variance.

    """
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return 1 - np.exp(sigma * np.log(.5) / self._tol)
        else:
            return sigma / (sigma + self._tol)


class VarianceDecreasingParameter(VarianceParameter):
    """
    Class implementing a parameter that decreases with the target
    variance.

    """
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return np.exp(sigma * np.log(.5) / self._tol)
        else:
            return 1. / (sigma + self._tol)


class WindowedVarianceParameter(Parameter):
    """
    Abstract class to implement variance-dependent parameters. A ``target``
    parameter is expected. differently from the "Variance Parameter" class
    the variance is computed in a window interval.

    """
    def __init__(self, value, exponential=False, min_value=None, tol=1.,
                 window=100, size=(1,)):
        """
        Constructor.

        Args:
            tol (float): value of the variance of the target variable such that the
                parameter value is 0.5.
            window (int):
        """
        self._exponential = exponential
        self._tol = tol
        self._weights_var = Table(size)
        self._samples = Table(size + (window,))
        self._index = Table(size, dtype=int)
        self._window = window
        self._parameter_value = Table(size)

        self._add_save_attr(
            _exponential='primitive',
            _tol='primitive',
            _weights_var='mushroom',
            _samples='mushroom',
            _index='mushroom',
            _window='primitive',
            _parameter_value='mushroom',
        )

        super(WindowedVarianceParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        return self._parameter_value[idx]

    def update(self, *idx, **kwargs):
        """
        Updates the value of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter whose number of visits has to be
                updated.
            target (float): Value of the target variable;
            factor (float): Multiplicative factor for the parameter value, useful
                when the parameter depend on another parameter value.

        """
        x = kwargs['target']
        factor = kwargs.get('factor', 1.)

        # compute parameter value
        n = self._n_updates[idx]
        self._n_updates[idx] += 1

        if n < 2:
            parameter_value = self._initial_value
        else:
            samples = self._samples[idx]

            if n < self._window:
                samples = samples[:int(n)]

            var = np.var(samples)
            var_estimator = var * self._weights_var[idx]
            parameter_value = self._compute_parameter(var_estimator,
                                                      sigma_process=var,
                                                      index=idx)

        # update state
        index = np.array([self._index[idx]], dtype=int)
        self._samples[idx + (index,)] = x
        self._index[idx] += 1
        if self._index[idx] >= self._window:
            self._index[idx] = 0

        self._weights_var[idx] = (
            1. - factor*parameter_value) ** 2 * self._weights_var[idx] + (
            factor * parameter_value) ** 2
        self._parameter_value[idx] = parameter_value

    def _compute_parameter(self, sigma, **kwargs):
        raise NotImplementedError(
            'WindowedVarianceParameter is an abstract class.')


class WindowedVarianceIncreasingParameter(WindowedVarianceParameter):
    """
    Class implementing a parameter that decreases with the target
    variance, where the variance is computed in a fixed length
    window.

    """
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return 1 - np.exp(sigma * np.log(.5) / self._tol)
        else:
            return sigma / (sigma + self._tol)
