from keras import backend as K
from keras.optimizers import Optimizer


class RMSpropGraves(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.00025, rho=.95, squared_rho=.95, epsilon=.01,
                 decay=0., **kwargs):
        super(RMSpropGraves, self).__init__(**kwargs)
        self.lr = K.variable(lr, name='lr')
        self.squared_rho = K.variable(squared_rho, name='squared_rho')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.iterations = K.variable(0., name='iterations')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        a_accumulators = [K.zeros(shape) for shape in shapes]
        b_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = a_accumulators + b_accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        for p, g, a, b in zip(params, grads, a_accumulators, b_accumulators):
            # update accumulator
            new_a = self.squared_rho * a + (1. - self.squared_rho) * K.square(g)
            new_b = self.rho * b + (1. - self.rho) * g
            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(b, new_b))
            new_p = p - lr * g / K.sqrt(new_a - K.square(new_b) + self.epsilon)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'squared_rho': float(K.get_value(self.squared_rho)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSpropGraves, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))