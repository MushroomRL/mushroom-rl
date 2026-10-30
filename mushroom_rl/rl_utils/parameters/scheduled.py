from mushroom_rl.core import ArrayBackend
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
    def __init__(self, value, threshold_value, n, shape=None, log_full=False, backend='numpy', device=None):
        """
        Constructor.

        Args:
            value ([float, Array]): initial value of the parameter, a scalar or a vector;
            threshold_value ([float, Array]): value reached by the parameter after ``n`` steps, a scalar if
                ``value`` is a scalar, a vector of the same shape otherwise;
            n (int): number of time steps needed to reach the threshold value;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value);
            backend (str, 'numpy'): array backend of the parameter;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        coeff = (threshold_value - value) / n

        if self._is_array(coeff):
            assert self._is_array(value) and self._is_array(threshold_value), \
                'value and threshold_value must be both scalars or both vectors.'
            input_backend = ArrayBackend.get_array_backend_from(coeff)
            min_value = input_backend.minimum(value, threshold_value)
            max_value = input_backend.maximum(value, threshold_value)
            super().__init__(value=value, min_value=min_value, max_value=max_value, shape=shape, log_full=log_full,
                             backend=backend, device=device)
        elif coeff >= 0:
            super().__init__(value=value, max_value=threshold_value, shape=shape, log_full=log_full, backend=backend,
                             device=device)
        else:
            super().__init__(value=value, min_value=threshold_value, shape=shape, log_full=log_full, backend=backend,
                             device=device)

        self._coeff = self._to_backend(coeff)

        self._add_save_attr(_coeff=self._value_serialization_method(self._coeff))

    def set_backend(self, backend, device=None):
        super().set_backend(backend, device)

        self._coeff = self._to_backend(self._coeff)

        self._add_save_attr(_coeff=self._value_serialization_method(self._coeff))

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
    def __init__(self, value, exp=1., min_value=None, max_value=None, shape=None, log_full=False, backend='numpy',
                 device=None):
        """
        Constructor.

        Args:
            value ([float, Array]): initial value of the parameter, a scalar or a vector;
            exp (float, 1.): exponent for the step decay;
            min_value ([float, Array], None): minimum value that the parameter can reach when
                decreasing;
            max_value ([float, Array], None): maximum value that the parameter can reach when
                increasing;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value);
            backend (str, 'numpy'): array backend of the parameter;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        self._exp = exp

        super().__init__(value=value, min_value=min_value, max_value=max_value, shape=shape, log_full=log_full,
                         backend=backend, device=device)

        self._add_save_attr(_exp='primitive')

    def _compute(self, *args, n, **kwargs):
        n = n.clip(1, None)

        return self._initial_value / n ** self._exp
