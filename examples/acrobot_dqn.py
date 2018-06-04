import os

import numpy as np
import tensorflow as tf

from mushroom.algorithms.value import DQN
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter


class SimpleNet:
    def __init__(self, name=None, **convnet_pars):
        self._name = name
        self._session = tf.Session()
        self._build(convnet_pars)

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
        self._session.run(self._train_step,
                          feed_dict={self._x: s,
                                     self._action: a.ravel().astype(np.uint8),
                                     self._target_q: np.reshape(q, (-1, 1))})

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
            for i in range(1): #FIXME remove
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
                [None, 1],
                name='target_q'
            )
            loss = 0.
            for i in range(1):
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

    @property
    def n_features(self):
        return self._features.shape[-1]


# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def experiment(n_epochs, n_steps, n_steps_test):
    np.random.seed()

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gym('Acrobot-v1', horizon, gamma)

    # Policy
    epsilon = Parameter(value=0.01)
    pi = EpsGreedy(epsilon=epsilon)

    # Settings
    initial_replay_size = 100
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = mdp.info.observation_space.shape + (1,)
    approximator_params = dict(n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               optimizer={'name': 'adam',
                                          'lr': .0001,
                                          'decay': .95,
                                          'epsilon': .01})


    # Agent
    agent = DQN(SimpleNet, pi, mdp.info,
                approximator_params=approximator_params,
                batch_size=batch_size,
                n_approximators=1,
                initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                history_length=1,
                target_update_frequency=target_update_frequency,
                max_no_op_actions=0,
                no_op_action_value=0,
                dtype=np.float32)


    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma)
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('Epoch: ', n)
        core.learn(n_steps=n_steps,
                   n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma)
        print('J: ', np.mean(J))

    print('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    experiment(n_epochs=50, n_steps=1000, n_steps_test=1000)
