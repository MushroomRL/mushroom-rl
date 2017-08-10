import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense
from keras.optimizers import Adam

from PyPi.algorithms.dqn import DQN
from PyPi.approximators import Regressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.dataset import compute_scores
from PyPi.utils.parameters import LinearDecayParameter, Parameter
from PyPi.utils.preprocessor import Scaler


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
        self.optimizer = Adam()

        def mean_squared_error_clipped(y_true, y_pred):
            return K.clip(K.mean(K.square(y_pred - y_true), axis=-1), -1, 1)

        # Compile
        self.q.compile(optimizer=self.optimizer,
                       loss=mean_squared_error_clipped)

    def fit(self, x, y, **fit_params):
        self.q.fit(x, y, **fit_params)

    def predict(self, x, **fit_params):
        if isinstance(x, list):
            assert len(x) == 2

            return self.q.predict(x, **fit_params)
        else:
            return self.all_q.predict(x, **fit_params)

    def train_on_batch(self, x, y, **fit_params):
        self.q.train_on_batch(x, y, **fit_params)

    def set_weights(self, w):
        self.q.set_weights(w)

    def get_weights(self):
        return self.q.get_weights()


def experiment():
    np.random.seed()
    scale_coeff = 10.

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
    mdp = Atari(mdp_name, train=True)
    # MDP test
    mdp_test = Atari(mdp_name)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)
    epsilon_test = Parameter(value=.05)
    pi_test = EpsGreedy(epsilon=epsilon_test,
                        observation_space=mdp.observation_space,
                        action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(n_actions=mdp.action_space.n)
    approximator = Regressor(ConvNet,
                             fit_action=False,
                             preprocessor=[Scaler(mdp.observation_space.high)],
                             **approximator_params)

    # Agent
    algorithm_params = dict(
        batch_size=32,
        target_approximator=Regressor(
            ConvNet,
            fit_action=False,
            preprocessor=[Scaler(mdp.observation_space.high)],
            **approximator_params),
        initial_dataset_size=initial_dataset_size,
        target_update_frequency=target_update_frequency)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DQN(approximator, pi, **agent_params)
    agent_test = DQN(approximator, pi_test, **agent_params)

    # Algorithm
    core = Core(agent, mdp, max_dataset_size=max_dataset_size)
    core_test = Core(agent_test, mdp_test)

    # DQN
    core.learn(n_iterations=evaluation_update_frequency, how_many=1,
               n_fit_steps=1, iterate_over='samples')
    core_test.evaluate(n_episodes=n_test_episodes)
    score = compute_scores(core_test.get_dataset())
    print('min_reward: %f, max_reward: %f, mean_reward: %f' % score)
    agent.policy.set_epsilon(LinearDecayParameter(value=1,
                                                  min_value=0.1,
                                                  n=final_exploration_frame))
    for i in xrange(max_steps - evaluation_update_frequency):
        core.learn(n_iterations=evaluation_update_frequency, how_many=1,
                   n_fit_steps=1, iterate_over='samples')
        core_test.reset()
        core_test.evaluate(n_episodes=n_test_episodes)
        score = compute_scores(core_test.get_dataset())
        print('min_reward: %f, max_reward: %f, mean_reward: %f' % score)

if __name__ == '__main__':
    logger.Logger(1)
    experiment()
