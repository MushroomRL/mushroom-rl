import numpy as np

from mushroom_rl.core.serialization import Serializable
from mushroom_rl.approximators.table import Table


def to_parameter(x):
    if isinstance(x, Parameter):
        return x
    else:
        return Parameter(x)


class Parameter(Serializable):
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.

    """
    def __init__(self, value, min_value=None, max_value=None, size=(1,)):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            min_value (float, None): minimum value that the parameter can reach when decreasing;
            max_value (float, None): maximum value that the parameter can reach when increasing;
            size (tuple, (1,)): shape of the matrix of parameters; this shape can be used to have a single parameter for
                each state or state-action tuple.

        """
        self._initial_value = value
        self._min_value = min_value
        self._max_value = max_value
        self._n_updates = Table(size)

        self._add_save_attr(
            _initial_value='primitive',
            _min_value='primitive',
            _max_value='primitive',
            _n_updates='mushroom',
        )

    def __call__(self, *idx, **kwargs):
        """
        Update and return the parameter in the provided index.

        Args:
             *idx (list): index of the parameter to return.

        Returns:
            The updated parameter in the provided index.

        """
        if self._n_updates.table.size == 1:
            idx = list()

        self.update(*idx, **kwargs)

        return self.get_value(*idx, **kwargs)

    def get_value(self, *idx, **kwargs):
        """
        Return the current value of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter to return.

        Returns:
            The current value of the parameter in the provided index.

        """
        new_value = self._compute(*idx, **kwargs)

        if self._min_value is None and self._max_value is None:
            return new_value
        else:
            return np.clip(new_value, self._min_value, self._max_value)

    def _compute(self, *idx, **kwargs):
        """
        Returns:
            The value of the parameter in the provided index.

        """
        return self._initial_value

    def update(self, *idx, **kwargs):
        """
        Updates the number of visit of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter whose number of visits has to be updated.

        """
        self._n_updates[idx] += 1

    @property
    def shape(self):
        """
        Returns:
            The shape of the table of parameters.

        """
        return self._n_updates.table.shape

    @property
    def initial_value(self):
        """
        Returns:
            The initial value of the parameters.

        """
        return self._initial_value


class LinearParameter(Parameter):
    r"""
    This class implements a linearly changing parameter according to the number of times it has been used.
    The parameter changes following the formula:

    .. math::
        v_n = \textrm{clip}(v_0 + \dfrac{v_{th} - v_0}{n}, v_{th})

    where :math:`v_0` is the initial value of the parameter,  :math:`n` is the number of steps and  :math:`v_{th}` is
    the upper or lower threshold for the parameter.

    """
    def __init__(self, value, threshold_value, n, size=(1,)):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            threshold_value (float, None): minimum or maximum value that the parameter can reach;
            n (int): number of time steps needed to reach the threshold value;
            size (tuple, (1,)): shape of the matrix of parameters; this shape can be used to have a single parameter for
                each state or state-action tuple.

        """
        self._coeff = (threshold_value - value) / n

        if self._coeff >= 0:
            super().__init__(value, None, threshold_value, size)
        else:
            super().__init__(value, threshold_value, None, size)

        self._add_save_attr(_coeff='primitive')

    def _compute(self, *idx, **kwargs):
        return self._coeff * self._n_updates[idx] + self._initial_value


class DecayParameter(Parameter):
    r"""
    This class implements a decaying parameter. The decay follows the formula:

    .. math::
        v_n = \dfrac{v_0}{n^p}

    where :math:`v_0` is the initial value of the parameter,  :math:`n` is the number of steps and  :math:`p` is an
    arbitrary exponent.

    """
    def __init__(self, value, exp=1., min_value=None, max_value=None, size=(1,)):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            exp (float, 1.): exponent for the step decay;
            min_value (float, None): minimum value that the parameter can reach when decreasing;
            max_value (float, None): maximum value that the parameter can reach when increasing;
            size (tuple, (1,)): shape of the matrix of parameters; this shape can be used to have a single parameter for
                each state or state-action tuple.

        """
        self._exp = exp

        super().__init__(value, min_value, max_value, size)

        self._add_save_attr(_exp='primitive')

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)

        return self._initial_value / n ** self._exp

