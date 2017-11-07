import numpy as np

from mushroom.utils.parameters import Parameter
from mushroom.utils.table import Table


class VarianceParameter(Parameter):
    """
    Abstract class to implement variance-dependent parameters. A `target`
    parameter is expected.

    """
    def __init__(self, value, exponential=False, min_value=None, tol=1.,
                 shape=(1,)):
        self._exponential = exponential
        self._tol = tol
        self._weights_var = Table(shape)
        self._x = Table(shape)
        self._x2 = Table(shape)
        self._parameter_value = Table(shape)

        super(VarianceParameter, self).__init__(value, min_value, shape)

    def _compute(self, *idx, **kwargs):
        return self._parameter_value[idx]

    def _update(self, *idx, **kwargs):
        x = kwargs['target']
        factor = kwargs.get('factor', 1.)

        # compute parameter value
        n = self._n_updates[idx] - 1

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
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return 1 - np.exp(sigma * np.log(.5) / self._tol)
        else:
            return sigma / (sigma + self._tol)


class VarianceDecreasingParameter(VarianceParameter):
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return np.exp(sigma * np.log(.5) / self._tol)
        else:
            return 1. / (sigma + self._tol)


class WindowedVarianceParameter(Parameter):
    def __init__(self, value, exponential=False, min_value=None, tol=1.,
                 window=100, shape=(1,)):
        self._exponential = exponential
        self._tol = tol
        self._weights_var = Table(shape)
        self._samples = Table(shape + (window,))
        self._index = Table(shape, dtype=int)
        self._window = window
        self._parameter_value = Table(shape)

        super(WindowedVarianceParameter, self).__init__(value, min_value, shape)

    def _compute(self, *idx, **kwargs):
        return self._parameter_value[idx]

    def _update(self, *idx, **kwargs):
        x = kwargs['target']
        factor = kwargs.get('factor', 1.)

        # compute parameter value
        n = self._n_updates[idx] - 1

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
        self._samples[idx][self._index[idx]] = x
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
    def _compute_parameter(self, sigma, **kwargs):
        if self._exponential:
            return 1 - np.exp(sigma * np.log(.5) / self._tol)
        else:
            return sigma / (sigma + self._tol)
