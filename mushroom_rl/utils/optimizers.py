import numpy as np
import numpy_ml as npml


class Optimizer(object):
    """
    Base class for gradient optimizers.
    These objects take the current parameters and the gradient estimate to compute the new parameters.

    """

    def __init__(self, *params):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self,  *args, **kwargs):
        raise NotImplementedError


class AdaptiveParameterOptimizer(Optimizer):
    """
    This class implements an adaptive gradient step optimizer.
    Instead of moving of a step proportional to the gradient,
    takes a step limited by a given metric.
    To specify the metric, the natural gradient has to be provided. If natural
    gradient is not provided, the identity matrix is used.

    The step rule is:

    .. math::
        \\Delta\\theta=\\underset{\\Delta\\vartheta}{argmax}\\Delta\\vartheta^{t}\\nabla_{\\theta}J

        s.t.:\\Delta\\vartheta^{T}M\\Delta\\vartheta\\leq\\varepsilon

    Lecture notes, Neumann G.
    http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf

    """
    def __init__(self, value, maximize=True):
        """0
        Constructor.

        Args:
            value (float): the maximum step defined by the metric
            maximize (bool): by default Optimizers do a gradient ascent step. Set to False for gradient descent
        """
        super().__init__()
        self._eps = value
        self._maximize = maximize

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        lr = self.get_value(*args[1:], **kwargs)
        if not self._maximize:
            grads *= -1
        return params + lr * grads

    def get_value(self, *args, **kwargs):
        if len(args) == 2:
            gradient = args[0]
            nat_gradient = args[1]
            tmp = (gradient.dot(nat_gradient)).item()
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


class FixedLearningRateOptimizer(Optimizer):
    """
    This class implements a fixed learning rate optimizer.

    """
    def __init__(self, value, maximize=True):
        """
        Constructor.

        Args:
            value (float): the learning rate
            maximize (bool): by default Optimizers do a gradient ascent step. Set to False for gradient descent
        """
        super().__init__()
        self._lr = value
        self._maximize = maximize

    def __call__(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        return self.update(params, grads)

    def update(self, params, grads):
        if not self._maximize:
            grads *= -1
        return params + self._lr * grads


class AdamOptimizer(Optimizer):
    """
    This class implements the Adam optimizer.

    """
    def __init__(self, value, decay1=0.9, decay2=0.999, maximize=True):
        """
        Constructor.

        Args:
            value (float): the initial learning rate
            decay1 (float): Adam beta1 parameter
            decay2 (float): Adam beta2 parameter
            maximize (bool): by default Optimizers do a gradient ascent step. Set to False for gradient descent
        """
        super().__init__()
        self._optimizer = npml.neural_nets.optimizers.Adam(
            lr=value,
            decay1=decay1,
            decay2=decay2
        )
        self._maximize = maximize

    def __call__(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        return self.update(params, grads)

    def update(self, params, grads):
        if self._maximize:
            # -1*grads because numpy_ml does gradient descent by default, not ascent
            grads *= -1
        return self._optimizer.update(params, grads, 'theta')
