import math

import numpy as np

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.approximators.table import Table


class Parameter(MushroomObject):
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.

    """
    def __init__(self, value, shape=None, log_full=False):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value). By default, non-scalar parameters are not logged, as logging a per-state or
                per-state-action value on every update is too expensive.

        """
        self._initial_value = value
        self._shape = shape
        self._log_full = log_full

        self._add_save_attr(
            _initial_value='primitive',
            _shape='primitive',
            _log_full='primitive',
        )

    def __call__(self, *args, **kwargs):
        """
        Update and return the parameter at the provided point.

        Args:
             *args: point at which the parameter is evaluated (e.g. the state or state-action index for a
                tabular parameter; ignored for a scalar one).

        Returns:
            The updated parameter at the provided point.

        """
        self.update(*args, **kwargs)

        return self.get_value(*args, **kwargs)

    def get_value(self, *args, **kwargs):
        """
        Return the current value of the parameter at the provided point.

        Args:
            *args: point at which the parameter is evaluated (e.g. the state or state-action index for a
                tabular parameter; ignored for a scalar one).

        Returns:
            The current value of the parameter at the provided point.

        """
        return self._compute(*args, **kwargs)

    def update(self, *args, **kwargs):
        """
        Update the internal state of the parameter at the provided point.

        Args:
            *args: point at which the parameter is updated (e.g. the state or state-action index for a
                tabular parameter; ignored for a scalar one).

        """
        self._log(*args, **kwargs)

    @staticmethod
    def make(x):
        """
        Args:
            x ([float, Parameter]): the value to convert to a parameter, if it is not one already.

        Returns:
            ``x`` unchanged if it is already a ``Parameter``, otherwise a constant ``Parameter`` wrapping it.

        """
        if isinstance(x, Parameter):
            return x
        else:
            return Parameter(x)

    def _compute(self, *args, **kwargs):
        """
        Returns:
            The value of the parameter at the provided point.

        """
        return self._initial_value

    def _log(self, *args, **kwargs):
        should_log = self._logger is not None and (self._log_full or math.prod(self.shape) == 1)
        if should_log:
            name = self._log_label or 'value'
            value = float(np.squeeze(self.get_value(*args, **kwargs)))
            self._logger.log_training(self._log_prefix, **{name: value})

    @property
    def shape(self):
        """
        Returns:
            The shape of the parameter.

        """
        return self._shape if self._shape is not None else (1,)

    @property
    def initial_value(self):
        """
        Returns:
            The initial value of the parameters.

        """
        return self._initial_value


class VariableParameter(Parameter):
    """
    Base class for parameters whose value depends on the number of times they have been used. Adds a per-index
    visit counter on top of :class:`Parameter`, used by the formula implemented by subclasses.

    """
    def __init__(self, value, min_value=None, max_value=None, shape=None, log_full=False):
        super().__init__(value, shape=shape, log_full=log_full)

        self._min_value = min_value
        self._max_value = max_value
        self._n_updates = self._make_approximator(shape if shape is not None else (1,))

        self._add_save_attr(
            _min_value='primitive',
            _max_value='primitive',
            _n_updates='mushroom',
        )

    def get_value(self, *args, **kwargs):
        new_value = super().get_value(*args, n=self._n_updates[args], **kwargs)

        if self._min_value is None and self._max_value is None:
            return new_value
        else:
            return np.clip(new_value, self._min_value, self._max_value)

    def update(self, *args, **kwargs):
        self._n_updates[args] += 1

        super().update(*args, **kwargs)

    def _make_approximator(self, shape, dtype=None):
        """
        Build the approximator backing this parameter's per-point statistics. Returns a :class:`Table` by
        default; override to use a different approximator.

        Args:
            shape (tuple): shape of the approximator;
            dtype ([int, float], None): dtype of the approximator.

        Returns:
            The approximator, a :class:`Table` by default.

        """
        return Table(shape, dtype=dtype)


class LinearParameter(VariableParameter):
    r"""
    This class implements a linearly changing parameter according to the number of times it has been used.
    The parameter changes following the formula:

    .. math::
        v_n = \textrm{clip}(v_0 + \dfrac{v_{th} - v_0}{n}, v_{th})

    where :math:`v_0` is the initial value of the parameter,  :math:`n` is the number of steps and  :math:`v_{th}` is
    the upper or lower threshold for the parameter.

    """
    def __init__(self, value, threshold_value, n, shape=None, log_full=False):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            threshold_value (float, None): minimum or maximum value that the parameter can reach;
            n (int): number of time steps needed to reach the threshold value;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value).

        """
        self._coeff = (threshold_value - value) / n

        if self._coeff >= 0:
            super().__init__(value=value, max_value=threshold_value, shape=shape, log_full=log_full)
        else:
            super().__init__(value=value, min_value=threshold_value, shape=shape, log_full=log_full)

        self._add_save_attr(_coeff='primitive')

    def _compute(self, *args, n, **kwargs):
        return self._coeff * n + self._initial_value


class DecayParameter(VariableParameter):
    r"""
    This class implements a decaying parameter. The decay follows the formula:

    .. math::
        v_n = \dfrac{v_0}{n^p}

    where :math:`v_0` is the initial value of the parameter,  :math:`n` is the number of steps and  :math:`p` is an
    arbitrary exponent.

    """
    def __init__(self, value, exp=1., min_value=None, max_value=None, shape=None, log_full=False):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            exp (float, 1.): exponent for the step decay;
            min_value (float, None): minimum value that the parameter can reach when decreasing;
            max_value (float, None): maximum value that the parameter can reach when increasing;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value).

        """
        self._exp = exp

        super().__init__(value=value, min_value=min_value, max_value=max_value, shape=shape, log_full=log_full)

        self._add_save_attr(_exp='primitive')

    def _compute(self, *args, n, **kwargs):
        n = np.maximum(n, 1)

        return self._initial_value / n ** self._exp
