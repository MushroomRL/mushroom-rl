import numpy as np
import numpy_ml as npml

from mushroom_rl.core import Serializable
from mushroom_rl.utils.parameters import Parameter


class Optimizer(Serializable):
    """
    Base class for gradient optimizers.
    These objects take the current parameters and the gradient estimate to compute the new parameters.

    """
    def __init__(self, lr=0.001, maximize=True, *params):
        """
        Constructor

        Args:
            lr ([float, Parameter]): the learning rate;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        if isinstance(lr, float):
            self._lr = Parameter(lr)
        else:
            self._lr = lr
        self._maximize = maximize

        self._add_save_attr(
            _lr='mushroom',
            _maximize='primitive'
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AdaptiveOptimizer(Optimizer):
    """
    This class implements an adaptive gradient step optimizer.
    Instead of moving of a step proportional to the gradient,
    takes a step limited by a given metric M.
    To specify the metric, the natural gradient has to be provided. If natural
    gradient is not provided, the identity matrix is used.

    The step rule is:

    .. math::
        \\Delta\\theta=\\underset{\\Delta\\vartheta}{argmax}\\Delta\\vartheta^{t}\\nabla_{\\theta}J

        s.t.:\\Delta\\vartheta^{T}M\\Delta\\vartheta\\leq\\varepsilon

    Lecture notes, Neumann G.
    http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf

    """
    def __init__(self, eps, maximize=True):
        """
        Constructor.

        Args:
            eps (float): the maximum step defined by the metric;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(maximize=maximize)
        self._eps = eps

        self._add_save_attr(_eps='primitive')

    def __call__(self, params, *args, **kwargs):
        # If two args are passed
        # args[0] is the gradient g, and grads[1] is the natural gradient M^{-1}g
        grads = args[0]
        if len(args) == 2:
            grads = args[1]
        lr = self.get_value(*args, **kwargs)
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


class SGDOptimizer(Optimizer):
    """
    This class implements the SGD optimizer.

    """
    def __init__(self, lr=0.001, maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)

    def __call__(self, params, grads):
        if not self._maximize:
            grads *= -1
        return params + self._lr() * grads


class AdamOptimizer(Optimizer):
    """
    This class implements the Adam optimizer.

    """
    def __init__(self, lr=0.001, decay1=0.9, decay2=0.999, maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            decay1 (float, 0.9): Adam beta1 parameter;
            decay2 (float, 0.999): Adam beta2 parameter;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)
        # lr_scheduler must be set to None, as we have our own scheduler
        self._optimizer = npml.neural_nets.optimizers.Adam(
            lr=self._lr.initial_value,
            decay1=decay1,
            decay2=decay2,
            lr_scheduler=None
        )

        self._add_save_attr(_optimizer='pickle')

    def __call__(self, params, grads):
        if self._maximize:
            # -1*grads because numpy_ml does gradient descent by default, not ascent
            grads *= -1
        # Fix the numpy_ml optimizer lr to the one we computed
        self._optimizer.lr_scheduler.lr = self._lr()
        return self._optimizer.update(params, grads, 'theta')


class AdaGradOptimizer(Optimizer):
    """
    This class implements the AdaGrad optimizer.

    """
    def __init__(self, lr=0.001, maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)
        # lr_scheduler must be set to None, as we have our own scheduler
        self._optimizer = npml.neural_nets.optimizers.AdaGrad(
            lr=self._lr.initial_value,
            lr_scheduler=None
        )

        self._add_save_attr(_optimizer='pickle')

    def __call__(self, params, grads):
        if self._maximize:
            # -1*grads because numpy_ml does gradient descent by default, not ascent
            grads *= -1
        # Fix the numpy_ml optimizer lr to the one we computed
        self._optimizer.lr_scheduler.lr = self._lr()
        return self._optimizer.update(params, grads, 'theta')


class RMSPropOptimizer(Optimizer):
    """
    This class implements the RMSProp optimizer.

    """
    def __init__(self, lr=0.001, decay=0.9, maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            decay (float, 0.9): rate of decay for the moving average;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)
        # lr_scheduler must be set to None, as we have our own scheduler
        self._optimizer = npml.neural_nets.optimizers.RMSProp(
            lr=self._lr.initial_value,
            decay=decay,
            lr_scheduler=None
        )

        self._add_save_attr(_optimizer='pickle')

    def __call__(self, params, grads):
        if self._maximize:
            # -1*grads because numpy_ml does gradient descent by default, not ascent
            grads *= -1
        # Fix the numpy_ml optimizer lr to the one we computed
        self._optimizer.lr_scheduler.lr = self._lr()
        return self._optimizer.update(params, grads, 'theta')
