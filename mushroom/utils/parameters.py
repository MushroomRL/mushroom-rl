from mushroom.utils.table import Table
import numpy as np


class Parameter(object):
    def __init__(self, value, min_value=None, shape=(1,)):
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = Table(shape)

    def __call__(self, *idx, **kwargs):
        if self._n_updates.table.size == 1:
            idx = []

        self.update(*idx)

        return self.get_value(*idx, **kwargs)

    def get_value(self, *idx, **kwargs):
        new_value = self._compute(*idx, **kwargs)

        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, *idx, **kwargs):
        return self._initial_value

    def update(self, *idx, **kwargs):
        self._n_updates[idx] += 1

    @property
    def shape(self):
        return self._n_updates.table.shape


class LinearDecayParameter(Parameter):
    def __init__(self, value,  min_value, n, shape=(1,)):
        self._coeff = (min_value - value) / n

        super(LinearDecayParameter, self).__init__(value, min_value, shape)

    def _compute(self, *idx, **kwargs):
        return self._coeff * self._n_updates[idx] + self._initial_value


class ExponentialDecayParameter(Parameter):
    def __init__(self, value, decay_exp=1., min_value=None, shape=(1,)):
        self._decay_exp = decay_exp

        super(ExponentialDecayParameter, self).__init__(value, min_value, shape)

    def _compute(self, *idx, **kwargs):
        return self._initial_value / self._n_updates[idx] ** self._decay_exp


class AdaptiveParameter(object):
    def __init__(self, value):
        self._eps = value

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)

    def get_value(self, *args, **kwargs):
        if len(args) == 2:
            gradient = args[0]
            nat_gradient = args[1]
            tmp = np.asscalar(np.dot(gradient.T, nat_gradient))
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
