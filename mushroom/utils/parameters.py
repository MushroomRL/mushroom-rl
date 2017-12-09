from mushroom.utils.table import Table
import numpy as np


class Parameter(object):
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.

    """
    def __init__(self, value, min_value=None, size=(1,)):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            min_value (float): minimum value that it can reach when decreasing;
            size (tuple): shape of the matrix of parameters; this shape can be
                used to have a single parameter for each state or state-action
                tuple.

        """
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = Table(size)

    def __call__(self, *idx, **kwargs):
        """
        Update and return the parameter in the provided index.

        Args:
             *idx (list): index of the parameter to return.

        Returns:
            the updated parameter in the provided index.

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
            the current value of the parameter in the provided index.

        """
        new_value = self._compute(*idx, **kwargs)

        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, *idx, **kwargs):
        """
        Returns:
            the value of the parameter in the provided index.

        """
        return self._initial_value

    def update(self, *idx, **kwargs):
        """
        Updates the number of visit of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter whose number of visits has to be
                updated.

        """
        self._n_updates[idx] += 1

    @property
    def shape(self):
        """
        Returns:
            the shape of the table of parameters.

        """
        return self._n_updates.table.shape


class LinearDecayParameter(Parameter):
    """
    This class implements a linearly decaying parameter according to the number
    of times it has been used.

    """
    def __init__(self, value,  min_value, n, size=(1,)):
        self._coeff = (min_value - value) / n

        super(LinearDecayParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        return self._coeff * self._n_updates[idx] + self._initial_value


class ExponentialDecayParameter(Parameter):
    """
    This class implements a exponentially decaying parameter according to the
    number of times it has been used.

    """
    def __init__(self, value, decay_exp=1., min_value=None, size=(1,)):
        self._decay_exp = decay_exp

        super(ExponentialDecayParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)
        return self._initial_value / n ** self._decay_exp


class AdaptiveParameter(object):
    """
    This class implements a basic adaptive gradient step. Instead of moving of
    a step proportional to the gradient, takes a step limited by a given metric.
    To specify the metric, the natural gradient has to be provided. If natural
    gradient is not provided, the identity matrix is used.

    The step rule is:

    .. math::
        \\Delta\\theta=\\underset{\\Delta\\vartheta}{argmax}\\Delta\\vartheta^{t}\\nabla_{\\theta}J

        s.t.:\\Delta\\vartheta^{T}M\\Delta\\vartheta\\leq\\varepsilon

    Lecture notes, Neumann G.
    http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf

    """
    def __init__(self, value):
        self._eps = value

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)

    def get_value(self, *args, **kwargs):
        if len(args) == 2:
            gradient = args[0]
            nat_gradient = args[1]
            tmp = np.asscalar(gradient.dot(nat_gradient))
            lambda_v = np.sqrt(tmp / (4. * self._eps))
            # For numerical stability
            lambda_v = max(lambda_v, 1e-8)
            step_length = 1. / (2. * lambda_v)

            return step_length
        elif len(args) == 1:
            return self.get_value(args[0], args[0], **kwargs)
        else:
            raise ValueError('Adaptive parameters needs gradient or gradient'
                             'and natural gradient')

    @property
    def shape(self):
        return None
