import numpy as np

from mushroom_rl.rl_utils.parameters.parameter import VariableParameter


class VarianceParameter(VariableParameter):
    """
    Abstract class to implement variance-dependent parameters. A ``target``
    parameter is expected.

    """
    def __init__(self, value, exponential=False, min_value=None, tol=1., shape=None, log_full=False):
        """
        Constructor.

        Args:
            tol (float): value of the variance of the target variable such that
                The parameter value is 0.5.

        """
        self._exponential = exponential
        self._tol = tol
        table_shape = shape if shape is not None else (1,)
        self._weights_var = self._make_approximator(table_shape)
        self._x = self._make_approximator(table_shape)
        self._x2 = self._make_approximator(table_shape)
        self._parameter_value = self._make_approximator(table_shape)

        super().__init__(value, min_value=min_value, shape=shape, log_full=log_full)

        self._add_save_attr(
            _exponential='primitive',
            _tol='primitive',
            _weights_var='mushroom',
            _x='mushroom',
            _x2='mushroom',
            _parameter_value='mushroom',
        )

    def _compute(self, *args, **kwargs):
        return self._parameter_value[args]

    def update(self, *args, target, factor=1., **kwargs):
        """
        Updates the value of the parameter at the provided point.

        Args:
            *args: point at which the parameter is updated (e.g. the state or state-action index for a
                tabular parameter; ignored for a scalar one);
            target (float): value of the target variable;
            factor (float, 1.): multiplicative factor for the parameter value, useful when the parameter
                depends on another parameter value.

        """
        x = target

        # compute parameter value
        n = self._n_updates[args]
        self._n_updates[args] += 1

        if n < 2:
            parameter_value = self._initial_value
        else:
            var = n * (self._x2[args] - self._x[args] ** 2) / (n - 1.)
            var_estimator = var * self._weights_var[args]
            parameter_value = self._compute_parameter(var_estimator, sigma_process=var, index=args)

        # update state
        self._x[args] += (x - self._x[args]) / self._n_updates[args]
        self._x2[args] += (x ** 2 - self._x2[args]) / self._n_updates[args]
        self._weights_var[args] = ((1. - factor * parameter_value) ** 2 * self._weights_var[args]
                                   + (factor * parameter_value) ** 2)
        self._parameter_value[args] = parameter_value

        self._log(*args, **kwargs)

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


class WindowedVarianceParameter(VariableParameter):
    """
    Abstract class to implement variance-dependent parameters. A ``target``
    parameter is expected. differently from the "Variance Parameter" class
    the variance is computed in a window interval.

    """
    def __init__(self, value, exponential=False, min_value=None, tol=1., window=100, shape=None, log_full=False):
        """
        Constructor.

        Args:
            tol (float): value of the variance of the target variable such that the
                parameter value is 0.5.
            window (int):
        """
        self._exponential = exponential
        self._tol = tol
        table_shape = shape if shape is not None else (1,)
        self._weights_var = self._make_approximator(table_shape)
        self._samples = self._make_approximator(table_shape + (window,))
        self._index = self._make_approximator(table_shape, dtype=int)
        self._window = window
        self._parameter_value = self._make_approximator(table_shape)

        self._add_save_attr(
            _exponential='primitive',
            _tol='primitive',
            _weights_var='mushroom',
            _samples='mushroom',
            _index='mushroom',
            _window='primitive',
            _parameter_value='mushroom',
        )

        super().__init__(value, min_value=min_value, shape=shape, log_full=log_full)

    def _compute(self, *args, **kwargs):
        return self._parameter_value[args]

    def update(self, *args, target, factor=1., **kwargs):
        """
        Updates the value of the parameter at the provided point.

        Args:
            *args: point at which the parameter is updated (e.g. the state or state-action index for a
                tabular parameter; ignored for a scalar one);
            target (float): value of the target variable;
            factor (float, 1.): multiplicative factor for the parameter value, useful when the parameter
                depends on another parameter value.

        """
        x = target

        # compute parameter value
        n = self._n_updates[args]
        self._n_updates[args] += 1

        if n < 2:
            parameter_value = self._initial_value
        else:
            samples = self._samples[args]

            if n < self._window:
                samples = samples[:int(n)]

            var = np.var(samples)
            var_estimator = var * self._weights_var[args]
            parameter_value = self._compute_parameter(var_estimator, sigma_process=var, index=args)

        # update state
        index = np.array([self._index[args]], dtype=int)
        self._samples[args + (index,)] = x
        self._index[args] += 1
        if self._index[args] >= self._window:
            self._index[args] = 0

        self._weights_var[args] = ((1. - factor*parameter_value) ** 2 * self._weights_var[args] +
                                   (factor * parameter_value) ** 2)
        self._parameter_value[args] = parameter_value

        self._log(*args, **kwargs)

    def _compute_parameter(self, sigma, **kwargs):
        raise NotImplementedError('WindowedVarianceParameter is an abstract class.')


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
