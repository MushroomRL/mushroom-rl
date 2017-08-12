import os
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense
from keras.optimizers import Optimizer, Adam
from keras.losses import mean_squared_error, mean_absolute_error

from PyPi.algorithms.dqn import DQN
from PyPi.approximators import Regressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.dataset import compute_scores
from PyPi.utils.parameters import LinearDecayParameter, Parameter
from PyPi.utils.preprocessor import Scaler

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class GatherLayer(Layer):
    def __init__(self, n_actions, **kwargs):
        self.n_actions = n_actions
        super(GatherLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GatherLayer, self).build(input_shape)

    def call(self, args, mask=None):
        return self.gather_layer(args, self.n_actions)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1

    @staticmethod
    def gather_layer(args, n_actions):
        full_output, indices = args

        idx_flat = tf.reshape(indices, [-1])
        idx_onehot = tf.one_hot(idx_flat, n_actions)
        out = tf.multiply(idx_onehot, full_output)
        out = tf.reduce_sum(out, 1)

        return out


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
            new_p = p - lr * g / (new_a - K.sqrt(new_b) + self.epsilon)

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


class ConvNet:
    def __init__(self, n_actions):
        # Build network
        self.input = Input(shape=(4, 84, 84))
        self.u = Input(shape=(1,), dtype='int32')

        self.hidden = Convolution2D(32, (8, 8), padding='valid',
                                    activation='relu', strides=(4, 4),
                                    data_format='channels_first')(self.input)

        self.hidden = Convolution2D(64, (4, 4), padding='valid',
                                    activation='relu', strides=(2, 2),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Convolution2D(64, (3, 3), padding='valid',
                                    activation='relu', strides=(1, 1),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(512, activation='relu')(self.hidden)
        self.output = Dense(n_actions, activation='linear')(self.features)
        self.gather = GatherLayer(n_actions)([self.output, self.u])

        # Models
        self.all_q = Model(outputs=[self.output], inputs=[self.input])
        self.q = Model(outputs=[self.gather], inputs=[self.input, self.u])

        # Optimization algorithm
        #self.optimizer = RMSpropGraves()
        self.optimizer = Adam()

        def mean_squared_error_clipped(y_true, y_pred):
            return K.minimum(mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred))

        # Compile
        self.q.compile(optimizer=self.optimizer,
                       loss=mean_squared_error_clipped)

        #Tensorboard
        self.writer = tf.summary.FileWriter('./logs')

    def fit(self, x, y, **fit_params):
        self.q.fit(x, y, **fit_params)

    def predict(self, x, **fit_params):
        if isinstance(x, list):
            assert len(x) == 2

            return self.q.predict(x, **fit_params)
        else:
            return self.all_q.predict(x, **fit_params)

    def train_on_batch(self, x, y, **fit_params):
        loss = self.q.train_on_batch(x, y, **fit_params)
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                     simple_value=loss), ])
        self.writer.add_summary(summary)

    def set_weights(self, w):
        self.q.set_weights(w)

    def get_weights(self):
        return self.q.get_weights()


def print_epoch(epoch):
    print '################################################################'
    print 'epoch: ', epoch
    print '----------------------------------------------------------------'


def experiment():
    np.random.seed()
    scale_coeff = 10.
    render = False
    quiet = False

    # DQN Parameters
    initial_dataset_size = int(5e4 / scale_coeff)
    target_update_frequency = int(1e4)
    max_dataset_size = int(1e6 / scale_coeff)
    evaluation_update_frequency = int(5e4)
    max_steps = int(5e6)
    final_exploration_frame = int(1e6)
    n_test_episodes = 30

    mdp_name = 'BreakoutDeterministic-v4'
    # MDP train
    mdp = Atari(mdp_name, ends_at_life=True)

    # Policy
    epsilon = LinearDecayParameter(value=1,
                                   min_value=0.1,
                                   n=final_exploration_frame)
    epsilon_test = Parameter(value=.05)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random,
                   observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(n_actions=mdp.action_space.n)
    approximator = Regressor(ConvNet,
                             fit_action=False,
                             preprocessor=[Scaler(mdp.observation_space.high)],
                             **approximator_params)

    # target approximatior
    target_approximator = Regressor(
        ConvNet,
        fit_action=False,
        preprocessor=[Scaler(mdp.observation_space.high)],
        **approximator_params)
    target_approximator.model.set_weights(approximator.model.get_weights())

    # Agent
    algorithm_params = dict(
        batch_size=32,
        target_approximator=target_approximator,
        initial_dataset_size=initial_dataset_size,
        target_update_frequency=target_update_frequency)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DQN(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp, max_dataset_size=max_dataset_size)
    core_test = Core(agent, mdp)

    # DQN

    # fill replay memory with random dataset
    print_epoch(0)
    core.learn(n_iterations=evaluation_update_frequency, how_many=1,
               n_fit_steps=1, iterate_over='samples', quiet=quiet)

    # evaluate initial policy
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(ends_at_life=False)
    core_test.evaluate(n_episodes=n_test_episodes, render=render, quiet=quiet)
    score = compute_scores(core_test.get_dataset())

    print('min_reward: %f, max_reward: %f, mean_reward: %f' % score)
    for i in xrange(max_steps - evaluation_update_frequency):
        print_epoch(i+1)
        print '- Learning:'
        # learning step
        pi.set_epsilon(epsilon)
        mdp.set_episode_end(ends_at_life=True)
        core.learn(n_iterations=evaluation_update_frequency, how_many=1,
                   n_fit_steps=1, iterate_over='samples', quiet=quiet)
        print '- Evaluation:'
        # evaluation step
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(ends_at_life=False)
        core_test.reset()
        core_test.evaluate(n_episodes=n_test_episodes, render=render, quiet=quiet)
        score = compute_scores(core_test.get_dataset())
        print('min_reward: %f, max_reward: %f, mean_reward: %f' % score)

if __name__ == '__main__':
    logger.Logger(1)
    experiment()
