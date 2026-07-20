import math

import numpy as np

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.approximators.table import Table


class Parameter(MushroomObject):
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.

    """
    def __init__(self, value, shape=None, log_full=False, backend='numpy'):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            shape (tuple, None): shape of the matrix of parameters; this shape can be used to have a single
                parameter for each state or state-action tuple. If None, the parameter is a scalar;
            log_full (bool, False): if True, the parameter is logged even when it is non-scalar (it holds more
                than one value). By default, non-scalar parameters are not logged, as logging a per-state or
                per-state-action value on every update is too expensive;
            backend (str, 'numpy'): array backend the parameter's inputs are given in; the inputs are
                converted to numpy before indexing a non-scalar parameter.

        """
        self._initial_value = value
        self._shape = shape
        self._log_full = log_full
        self._backend = ArrayBackend.get_array_backend(backend)

        self._add_save_attr(
            _initial_value='primitive',
            _shape='primitive',
            _log_full='primitive',
            _backend='primitive',
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

    def set_backend(self, backend):
        """
        Set the array backend the parameter's inputs are given in.

        Args:
            backend (str): name of the array backend.

        """
        self._backend = ArrayBackend.get_array_backend(backend)

    @staticmethod
    def make(x, backend='numpy'):
        """
        Args:
            x ([float, Parameter]): the value to convert to a parameter, if it is not one already;
            backend (str, 'numpy'): array backend to assign to the parameter.

        Returns:
            ``x`` with its backend set to ``backend`` if it is already a ``Parameter``, otherwise a constant
            ``Parameter`` wrapping it in the given backend.

        """
        if isinstance(x, Parameter):
            x.set_backend(backend)
            return x
        else:
            return Parameter(x, backend=backend)

    def _compute(self, *args, **kwargs):
        """
        Returns:
            The value of the parameter at the provided point.

        """
        return self._initial_value

    def _to_numpy(self, args):
        """
        Convert the indexing inputs to numpy, the backend of the underlying storage. Scalar parameters ignore
        the index and are returned untouched. Override (together with ``_make_approximator``) to back the
        parameter with a store that expects a different backend, e.g. a neural network.

        """
        if self._is_scalar:
            return args

        return tuple(a if isinstance(a, np.ndarray) else self._backend.to_numpy(a) for a in args)

    def _log(self, *args, **kwargs):
        should_log = self._logger is not None and (self._log_full or self._is_scalar)
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
    def _is_scalar(self):
        """
        Returns:
            Whether the parameter holds a single value (so the index is ignored).

        """
        return math.prod(self.shape) == 1

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
    def __init__(self, value, min_value=None, max_value=None, shape=None, log_full=False, backend='numpy'):
        super().__init__(value, shape=shape, log_full=log_full, backend=backend)

        self._min_value = min_value
        self._max_value = max_value
        self._n_updates = self._make_approximator(shape if shape is not None else (1,))

        self._add_save_attr(
            _min_value='primitive',
            _max_value='primitive',
            _n_updates='mushroom',
        )

    def get_value(self, *args, **kwargs):
        args = self._to_numpy(args)
        new_value = super().get_value(*args, n=self._n_updates[args], **kwargs)

        if self._min_value is None and self._max_value is None:
            return new_value
        else:
            return np.clip(new_value, self._min_value, self._max_value)

    def update(self, *args, **kwargs):
        args = self._to_numpy(args)
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
