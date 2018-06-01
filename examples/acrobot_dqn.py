import argparse
import datetime
import os

import numpy as np

from mushroom.algorithms.value import DQN
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import LinearDecayParameter, Parameter

import numpy as np
import tensorflow as tf


class SimpleNet:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name

        self._session = tf.Session()

        if load_path is not None:
            self._load(load_path, convnet_pars)
        else:
            self._build(convnet_pars)

        if self._name == 'train':
            self._train_saver = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))
        elif self._name == 'target':
            w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)

            with tf.variable_scope(self._scope_name):
                self._target_w = list()
                self._w = list()
                with tf.variable_scope('weights_placeholder'):
                    for i in range(len(w)):
                        self._target_w.append(tf.placeholder(w[i].dtype,
                                                             shape=w[i].shape))
                        self._w.append(w[i].assign(self._target_w[i]))

    def predict(self, s):
        res = np.array(
                [self._session.run(self._q, feed_dict={self._x: s})])
        return np.squeeze(res, axis=(0, 1))

    def fit(self, s, a, q):
        summaries, _ = self._session.run(
            [self._merged, self._train_step],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: np.reshape(q, (-1, 1))}
        )
        if hasattr(self, '_train_writer'):
            self._train_writer.add_summary(summaries, self._train_count)

        self._train_count += 1

    def set_weights(self, weights):
        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=self._scope_name)
        assert len(w) == len(weights)

        for i in range(len(w)):
            self._session.run(self._w[i],
                              feed_dict={self._target_w[i]: weights[i]})

    def get_weights(self, only_trainable=False):
        if not only_trainable:
            w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)
        else:
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self._scope_name)

        return self._session.run(w)

    def save(self):
        self._train_saver.save(
            self._session,
            self._folder_name + '/' + self._scope_name[:-1] + '/' +
            self._scope_name[:-1]
        )

    def _load(self, path, convnet_pars):
        self._scope_name = 'train/'
        restorer = tf.train.import_meta_graph(
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1] +
            '.meta')
        restorer.restore(
            self._session,
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1]
        )
        self._restore_collection(convnet_pars)

    def _build(self, convnet_pars):
        with tf.variable_scope(None, default_name=self._name):
            self._scope_name = tf.get_default_graph().get_name_scope() + '/'

            with tf.variable_scope('State'):
                self._x = tf.placeholder(tf.float32,
                                         shape=[None] + list(
                                             convnet_pars['input_shape']),
                                         name='input')

            with tf.variable_scope('Action'):
                self._action = tf.placeholder('uint8', [None], name='action')

                action_one_hot = tf.one_hot(self._action,
                                            convnet_pars['output_shape'][0],
                                            name='action_one_hot')

            x = self._x[..., 0]

            self._features = list()
            self._features2 = list()
            self._q = list()
            self._q_acted = list()
            for i in range(convnet_pars['n_approximators']):
                with tf.variable_scope('head_' + str(i)):
                    self._features.append(tf.layers.dense(
                        x, convnet_pars['n_features'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features_' + str(i)
                    ))
                    self._features2.append(tf.layers.dense(
                        self._features[i], convnet_pars['n_features'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features2_' + str(i)
                    ))
                    self._q.append(tf.layers.dense(
                        self._features2[i],
                        convnet_pars['output_shape'][0],
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='q_' + str(i)
                    ))
                    self._q_acted.append(
                        tf.reduce_sum(self._q[i] * action_one_hot,
                                      axis=1,
                                      name='q_acted_' + str(i))
                    )

            self._target_q = tf.placeholder(
                'float32',
                [None, convnet_pars['n_approximators']],
                name='target_q'
            )
            loss = 0.
            for i in range(convnet_pars['n_approximators']):
                loss += tf.losses.mean_squared_error(self._target_q[:, i],
                                                     self._q_acted[i])
            tf.summary.scalar('mse', loss)
            tf.summary.scalar('average_q', tf.reduce_mean(self._q))
            self._merged = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES,
                                  scope=self._scope_name)
            )

            optimizer = convnet_pars['optimizer']
            if optimizer['name'] == 'rmspropcentered':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=optimizer['lr'])
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=optimizer['lr'])
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_step = opt.minimize(loss=loss)

            initializer = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))

        self._session.run(initializer)

        if self._folder_name is not None:
            self._train_writer = tf.summary.FileWriter(
                self._folder_name + '/' + self._scope_name[:-1],
                graph=tf.get_default_graph()
            )

        self._train_count = 0

        self._add_collection()

    @property
    def n_features(self):
        return self._features.shape[-1]

    def _add_collection(self):
        tf.add_to_collection(self._scope_name + '_x', self._x)
        tf.add_to_collection(self._scope_name + '_action', self._action)
        for i in range(len(self._features)):
            tf.add_to_collection(self._scope_name + '_features_' + str(i),
                                 self._features[i])
            tf.add_to_collection(self._scope_name + '_features2_' + str(i),
                                 self._features2[i])
            tf.add_to_collection(self._scope_name + '_q_' + str(i), self._q[i])
            tf.add_to_collection(self._scope_name + '_q_acted_' + str(i),
                                 self._q_acted[i])
        tf.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self, convnet_pars):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]

        features = list()
        features2 = list()
        q = list()
        q_acted = list()
        for i in range(convnet_pars['n_approximators']):
            features.append(tf.get_collection(
                self._scope_name + '_features_' + str(i))[0])
            features2.append(tf.get_collection(
                self._scope_name + '_features2_' + str(i))[0])
            q.append(tf.get_collection(self._scope_name + '_q_' + str(i))[0])
            q_acted.append(tf.get_collection(
                self._scope_name + '_q_acted_' + str(i))[0])

        self._features = features
        self._features2 = features2
        self._q = q
        self._q_acted = q_acted
        self._target_q = tf.get_collection(self._scope_name + '_target_q')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = compute_scores(dataset)
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score))

    return score


def experiment():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_mdp = parser.add_argument_group('Environment')
    arg_mdp.add_argument("--horizon", type=int, default=200)
    arg_mdp.add_argument("--gamma", type=float, default=0.99)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--n-features", type=int, default=80)
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.0001,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered')
    arg_net.add_argument("--epsilon", type=float, default=.01,
                         help='Epsilon term used in rmspropcentered')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--n-approximators", type=int, default=1,
                         help="Number of approximators used in the ensemble for"
                              "Averaged DQN.")
    arg_alg.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=1,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=1000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=1,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=0.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=0.,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=1000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=0,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')
    arg_alg.add_argument("--no-op-action-value", type=int, default=0,
                         help='Value of the no-op action.')
    arg_alg.add_argument("--p-mask", type=float, default=1.)

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    scores = list()

    # Evaluation of the model provided by the user.
    if args.load_path:
        # MDP
        mdp = Gym('Acrobot-v1', args.horizon, args.gamma)
        # Policy
        epsilon_test = Parameter(value=args.test_exploration_rate)
        pi = EpsGreedy(epsilon=epsilon_test)

        # Approximator
        input_shape = mdp.info.observation_space.shape + (1,)
        approximator_params = dict(
            n_features=args.n_features,
            n_approximators=args.n_approximators,
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            name='test',
            load_path=args.load_path,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )

        approximator = SimpleNet

        # Agent
        algorithm_params = dict(
            batch_size=1,
            train_frequency=1,
            target_update_frequency=1,
            initial_replay_size=0,
            max_replay_size=0,
            history_length=args.history_length,
            max_no_op_actions=args.max_no_op_actions,
            no_op_action_value=args.no_op_action_value,
            dtype=np.float32
        )
        agent = DQN(approximator, pi, mdp.info,
                    approximator_params=approximator_params, **algorithm_params)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        pi.set_epsilon(epsilon_test)
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset)
    else:
        # DQN learning run

        # Summary folder
        folder_name = './logs/atari_' \
                      + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Settings
        if args.debug:
            initial_replay_size = 50
            max_replay_size = 500
            train_frequency = 5
            target_update_frequency = 10
            test_samples = 20
            evaluation_frequency = 50
            max_steps = 1000
        else:
            initial_replay_size = args.initial_replay_size
            max_replay_size = args.max_replay_size
            train_frequency = args.train_frequency
            target_update_frequency = args.target_update_frequency
            test_samples = args.test_samples
            evaluation_frequency = args.evaluation_frequency
            max_steps = args.max_steps

        # MDP
        mdp = Gym('Acrobot-v1', args.horizon, args.gamma)

        # Policy
        epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                       min_value=args.final_exploration_rate,
                                       n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1)
        pi = EpsGreedy(epsilon=epsilon_random)

        # Approximator
        input_shape = mdp.info.observation_space.shape + (1,)
        approximator_params = dict(
            n_features=args.n_features,
            n_approximators=args.n_approximators,
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            folder_name=folder_name,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )

        approximator = SimpleNet

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            n_approximators=args.n_approximators,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            history_length=args.history_length,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            max_no_op_actions=args.max_no_op_actions,
            no_op_action_value=args.no_op_action_value,
            dtype=np.float32
        )


        agent = DQN(approximator, pi, mdp.info,
                    approximator_params=approximator_params,
                    **algorithm_params)


        # Algorithm
        core = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.approximator.model.save()

        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        scores.append(get_stats(dataset))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            pi.set_epsilon(epsilon)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                    quiet=args.quiet)
            scores.append(get_stats(dataset))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
