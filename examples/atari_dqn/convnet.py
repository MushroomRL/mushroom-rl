import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class ConvNet:
    def __init__(self, n_actions, optimizer, name, width=84, height=84,
                 history_length=4):
        self._name = name
        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32,
                                     shape=[None,
                                            height,
                                            width,
                                            history_length],
                                     name='input')
            hidden_1 = tf.layers.conv2d(
                self._x, 32, 8, 4, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                bias_initializer=tf.glorot_uniform_initializer(),
                name='hidden_1'
            )
            hidden_2 = tf.layers.conv2d(
                hidden_1, 64, 4, 2, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                bias_initializer=tf.glorot_uniform_initializer(),
                name='hidden_2'
            )
            hidden_3 = tf.layers.conv2d(
                hidden_2, 64, 3, 1, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                bias_initializer=tf.glorot_uniform_initializer(),
                name='hidden_3'
            )
            flatten = tf.reshape(hidden_3, [-1, 7 * 7 * 64], name='flatten')
            features = tf.layers.dense(
                flatten, 512, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                bias_initializer=tf.glorot_uniform_initializer(),
                name='features'
            )
            self.q = tf.layers.dense(
                features, n_actions,
                kernel_initializer=tf.glorot_uniform_initializer(),
                bias_initializer=tf.glorot_uniform_initializer(),
                name='q'
            )

            self._target_q = tf.placeholder('float32', [None], name='target_q')
            self._action = tf.placeholder('uint8', [None], name='action')

            with tf.name_scope('gather'):
                action_one_hot = tf.one_hot(self._action, n_actions,
                                            name='action_one_hot')
                self._q_acted = tf.reduce_sum(self.q * action_one_hot,
                                              axis=1,
                                              name='q_acted')

            self._loss = tf.losses.huber_loss(self._target_q, self._q_acted)
            tf.summary.scalar('huber_loss', self._loss)

            if optimizer['name'] == 'rmspropgraves':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer()
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer()
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_count = 0
            self._train_step = opt.minimize(loss=self._loss)

        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

        self._merged = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter('./logs',
                                                   graph=tf.get_default_graph())

    def predict(self, x, **fit_params):
        if isinstance(x, list):
            return self._session.run(
                self._q_acted, feed_dict={self._x: x[0],
                                          self._action: x[1].ravel().astype(
                                              np.uint8)})
        return self._session.run(self.q, feed_dict={self._x: x})

    def train_on_batch(self, x, y, **fit_params):
        summaries, _ = self._session.run([self._merged, self._train_step],
                                         feed_dict={self._x: x[0],
                                         self._action: x[1].ravel().astype(
                                             np.uint8),
                                         self._target_q: y})
        self._train_writer.add_summary(summaries, self._train_count)

        self._train_count += 1

    def set_weights(self, weights):
        with tf.variable_scope(self._name):
            w = tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self._name)
            assert len(w) == len(weights)

            for i in xrange(len(w)):
                self._session.run(tf.assign(w[i], weights[i]))

    def get_weights(self):
        w = tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                              scope=self._name)

        return self._session.run(w)

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass


class ConvNetKeras:
    def __init__(self, n_actions, optimizer, width=84, height=84,
                 history_length=4):
        from keras.models import Model
        from keras.layers import Input, Convolution2D, Flatten, Dense
        # Build network
        input_layer = Input(shape=(height, width, history_length))

        hidden = Convolution2D(32, 8, padding='valid',
                               activation='relu', strides=4,
                               data_format='channels_last')(input_layer)

        hidden = Convolution2D(64, 4, padding='valid',
                               activation='relu', strides=2,
                               data_format='channels_last')(hidden)

        hidden = Convolution2D(64, 3, padding='valid',
                               activation='relu', strides=1,
                               data_format='channels_last')(hidden)

        hidden = Flatten()(hidden)
        features = Dense(512, activation='relu')(hidden)
        output = Dense(n_actions, activation='linear')(features)

        # Models
        self.q = Model(outputs=[output], inputs=[input_layer])

        def mean_squared_error_clipped(y_true, y_pred):
            return tf.where(tf.abs(y_true - y_pred) < 1.,
                            tf.square(y_true - y_pred) / 2.,
                            tf.abs(y_true - y_pred))

        # Compile
        self.q.compile(optimizer=optimizer,
                       loss=mean_squared_error_clipped)

        #Tensorboard
        self.writer = tf.summary.FileWriter('./logs')

    def predict(self, x, **fit_params):
        return self.q.predict(x, **fit_params)

    def train_on_batch(self, x, y, **fit_params):
        actions = x[1].astype(np.int)

        t = self.q.predict(x[0])
        for i in xrange(t.shape[0]):
            t[i, actions[i]] = y[i]

        loss = self.q.train_on_batch(x[0], t, **fit_params)
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                     simple_value=loss), ])
        self.writer.add_summary(summary)

    def save_weights(self, path):
        self.q.save_weights(path)

    def load_weights(self, path):
        self.q.load_weights(path)

    def set_weights(self, w):
        self.q.set_weights(w)

    def get_weights(self):
        return self.q.get_weights()
