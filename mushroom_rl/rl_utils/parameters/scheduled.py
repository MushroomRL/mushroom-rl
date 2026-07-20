import numpy as np

from mushroom_rl.rl_utils.parameters.parameter import VariableParameter


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
