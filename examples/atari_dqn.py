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
    def __init__(self, output_dim, nb_actions, **kwargs):
        """
        This layer can be used to split the output of the previous layer into
        nb_actions groups of size output_dim, and selectively choose which
        group to provide as output.
        It requires two tensors to be passed as input, namely the output of
        the previous layer and a column tensor with int32 or int64 values.
        Both inputs must obviously have the same batch_size.

        The input to this layer must be of shape (None, prev_output_dim),
        where prev_output_dim = output_dim * nb_actions.
        No checks are done at runtime to ensure that the input to the layer is
        correct, so you'd better double check.

        An example usage of this layer may be:
            input = Input(shape=(3,))
            control = Input(shape=(1,), dtype='int32')
            hidden = Dense(2 * 3)(i)  # output_dim == 2, nb_actions == 3
            output = GatherLayer(2, 3)([hidden, control])
            model = Model(input=[i, u], output=output)
            ...
            # Output is the first two neurons of hidden
            model.predict([randn(3), array([0])])
            # Output is the middle two neurons of hidden
            model.predict([randn(3), array([1])])
            # Output is the last two neurons of hidden
            model.predict([randn(3), array([2])])


        """
        self.output_dim = output_dim
        self.nb_actions = nb_actions
        super(GatherLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GatherLayer, self).build(input_shape)

    def call(self, args, mask=None):
        return self.gather_layer(args, self.output_dim, self.nb_actions)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    @staticmethod
    def gather_layer(args, output_size, nb_actions):
        full_output, indices = args
        '''
        Returns a tensor of shape (None, output_size) where each sample is
        the result of masking the corresponding sample in full_output with
        a binary mask that preserves only output_size elements, based on
         the corresponding index sample in indices.

        For example, given:
            full output: [[1, 2, 3, 4, 5, 6], [21, 22, 23, 24, 25, 26]]
            nb_actions: 3
            output_size: 2
            indices: [[2], [0]]
            desired output: [[5, 6], [21, 22]]
        we want the couple of elements [5, 6] representing the output
        for the third action (2) of the first sample, and [21, 22] representing
        the output for the first action (0) of the second sample;
        so we need the absolute indices [[4, 5], [0, 1]].

        To build these, we compute the first absolute indices (4 and 0) by
        multiplying the action indices for the output size:
            [[2], [0]] * 2 = [[4], [0]]

        '''
        base_absolute_indices = tf.multiply(indices, output_size)

        '''
        We then build an array containing the first absolute indices repeated
        output_size times:
            [[4, 4], [0, 0]]
        '''
        bai_repeated = tf.tile(base_absolute_indices, [1, output_size])

        '''
        Finally, we add range(output_size) to these tensors to get the full
        absolute indices tensors:
            [4, 4] + [0, 1] = [4, 5]
            [0, 0] + [0, 1] = [0, 1]
        so we get:
            [[4, 5], [0, 1]]
        '''
        absolute_indices = tf.add(bai_repeated, tf.range(output_size))

        '''
        We now need to flatten this tensor in order to later compute the
        one hot encoding for each absolute index:
            [4, 5, 0, 1]
        '''
        ai_flat = tf.reshape(absolute_indices, [-1])

        '''
        Compute the one-hot encoding for the absolute indices.
        Continuing the last example, from [4, 5, 0, 1] we now get:
            [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        '''
        ai_onehot = tf.one_hot(ai_flat, output_size * nb_actions)

        '''
        Build the mask for full_output from the onehot-encoded absolute indices.
        We need to group the one-hot absolute indices tensor into
        output_size-dimensional sub-tensors, in order to reduce_sum along
        axis 1 and get the correct masks.
        Therefore we get:
            [
              [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
              [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
            ]
        '''
        group_shape = [-1, output_size, output_size * nb_actions]
        group = tf.reshape(ai_onehot, group_shape)

        '''
        And with the reduce_sum along axis 1 we get:
            [[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]]
        '''
        masks = tf.reduce_sum(group, axis=1)

        '''
        Convert the mask to boolean. We get:
            [[False, False, False, False, True, True],
             [True, True, False, False, False, False]]
        '''
        zero = tf.constant(0, dtype=tf.float32)
        bool_masks = tf.not_equal(masks, zero)

        '''
        Convert the boolean mask to absolute indices for the full_output
        tensor (each element can be interpreted as [sample index, value index]).
        We get:
            [[0, 4], [0, 5], [1, 0], [1, 1]]
        '''
        ai_mask = tf.where(bool_masks)

        '''
        Apply the mask to the full output.
        We get a mono-dimensional tensor:
            [5, 6, 21, 22]
        '''
        reduced_output = tf.gather_nd(full_output, ai_mask)

        '''
        Reshape the reduction to match the output shape.
        We get:
            [[5, 6], [21, 22]]
        '''
        return tf.reshape(reduced_output, [-1, output_size])


class ConvNet:
    def __init__(self, n_actions):
        # Build network
        self.input = Input(shape=(4, 110, 84))
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
        self.gather = GatherLayer(1, n_actions)([self.output, self.u])

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
    initial_dataset_size = int(5e2 / scale_coeff)
    target_update_frequency = int(1e4)
    max_dataset_size = int(1e6 / scale_coeff)
    evaluation_update_frequency = int(5e4)
    max_steps = int(5e6)
    final_exploration_frame = int(1e6)
    n_test_episodes = 30

    mdp_name = 'BreakoutDeterministic-v3'
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
