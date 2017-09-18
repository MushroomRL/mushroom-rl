import numpy as np
import tensorflow as tf


class Extractor:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name
        self._n_features = convnet_pars.get('n_features', 512)
        self._reg_coeff = convnet_pars.get('reg_coeff', 0.)
        self._contractive = convnet_pars.get('contractive', False)
        self._predict_reward = convnet_pars.get('predict_reward', False)
        self._predict_absorbing = convnet_pars.get('predict_absorbing', False)

        self._session = tf.Session()

        if load_path is not None:
            self._load(load_path)
        else:
            self._build(convnet_pars)

        self._train_saver = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=self._scope_name))

    def predict(self, x, features=True, reconstruction=False, reward=False,
                absorbing=False):
        ret = list()
        if reconstruction:
            ret.append(self._session.run(self._predicted_frame,
                                         feed_dict={self._state: x[0],
                                                    self._action: x[1]}))
        if reward:
            ret.append(self._session.run(self._predicted_reward,
                                         feed_dict={self._state: x[0],
                                                    self._action: x[1]}))
        if absorbing:
            ret.append(self._session.run(self._predicted_absorbing,
                                         feed_dict={self._state: x[0],
                                                    self._action: x[1]}))
        if features:
            ret.append(self._session.run(self._features,
                                         feed_dict={self._state: x[0],
                                                    self._action: x[1]}))

        return ret

    def get_stats(self, x, y):
        f = [self._loss, self._xent, self._reg, self._xent_frame]
        fd = {self._state: x[0], self._action: x[1], self._target_frame: y[0]}

        if self._predict_reward:
            f += [self._xent_reward, self._accuracy_reward]
            fd[self._target_reward] = y[1]
        if self._predict_absorbing:
            f += [self._xent_absorbing, self._accuracy_absorbing]
            fd[self._target_absorbing] = y[2]
        stats = self._session.run(fetches=f, feed_dict=fd)
        stats_dict = dict(loss=stats[0],
                          xent=stats[1],
                          reg=stats[2],
                          xent_frame=stats[3]
                          )
        if self._predict_reward and self._predict_absorbing:
            stats_dict['xent_reward'] = stats[4]
            stats_dict['accuracy_reward'] = stats[5]
            stats_dict['xent_absorbing'] = stats[6]
            stats_dict['accuracy_absorbing'] = stats[7]
        elif self._predict_reward:
            stats_dict['xent_reward'] = stats[4]
            stats_dict['accuracy_reward'] = stats[5]
        elif self._predict_absorbing:
            stats_dict['xent_absorbing'] = stats[4]
            stats_dict['accuracy_absorbing'] = stats[5]

        return stats_dict

    def train_on_batch(self, x, y, **fit_params):
        fd = {self._state: x[0], self._action: x[1], self._target_frame: y}
        if self._predict_reward:
            fd[self._target_reward] = fit_params['target_reward']
        if self._predict_absorbing:
            fd[self._target_absorbing] = fit_params[
                'target_absorbing'].astype(np.int)

        summaries, _, self.loss = self._session.run(
            [self._merged, self._train_step, self._loss], feed_dict=fd)
        self._train_writer.add_summary(summaries, self._train_count)

        self._train_count += 1

    def save(self):
        self._train_saver.save(
            self._session,
            self._folder_name + '/' + self._scope_name + '/' + self._scope_name
        )

    def _load(self, path):
        restorer = tf.train.import_meta_graph(path + 'deep_fqi_extractor.meta')
        restorer.restore(self._session, path + 'deep_fqi_extractor')
        self._restore_collection()

    def _build(self, convnet_pars):
        with tf.variable_scope(self._name, default_name='deep_fqi_extractor'):
            self._scope_name = tf.get_default_graph().get_name_scope()
            self._state = tf.placeholder(tf.float32,
                                         shape=[None,
                                                convnet_pars['height'],
                                                convnet_pars['width'],
                                                convnet_pars['history_length']],
                                         name='state')
            self._action = tf.placeholder(tf.uint8,
                                          shape=[None, 1],
                                          name='action')
            one_hot_action = tf.one_hot(tf.reshape(self._action, [-1]),
                                        depth=convnet_pars['n_actions'],
                                        name='one_hot_action')
            hidden_1 = tf.layers.conv2d(
                self._state, 32, 8, 4, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_1'
            )
            hidden_2 = tf.layers.conv2d(
                hidden_1, 64, 4, 2, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_2'
            )
            hidden_3 = tf.layers.conv2d(
                hidden_2, 64, 3, 1, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_3'
            )
            hidden_4 = tf.layers.conv2d(
                hidden_3, 16, 3, 1, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_4'
            )
            hidden_4_flat = tf.reshape(hidden_4, [-1, 5 * 5 * 16],
                                       name='hidden_4_flat')
            features_state = tf.layers.dense(hidden_4_flat,
                                             self._n_features,
                                             activation=tf.nn.relu,
                                             name='features_state')
            features_action = tf.layers.dense(one_hot_action,
                                              self._n_features,
                                              activation=tf.nn.relu,
                                              name='features_action')
            state_x_action = tf.multiply(features_state, features_action,
                                         name='state_x_action')
            self._features = tf.layers.dense(state_x_action,
                                             self._n_features,
                                             activation=tf.nn.relu,
                                             name='features')
            hidden_5_flat = tf.layers.dense(self._features,
                                            400,
                                            activation=tf.nn.relu,
                                            name='hidden_5_flat')
            hidden_5_conv = tf.reshape(hidden_5_flat, [-1, 5, 5, 16],
                                       name='hidden_5_conv')
            hidden_5 = tf.layers.conv2d_transpose(
                hidden_5_conv, 16, 3, 1, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_5'
            )
            hidden_6 = tf.layers.conv2d_transpose(
                hidden_5, 64, 3, 1, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_6'
            )
            hidden_7 = tf.layers.conv2d_transpose(
                hidden_6, 64, 4, 2, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_7'
            )
            hidden_8 = tf.layers.conv2d_transpose(
                hidden_7, 32, 8, 4, activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='hidden_8'
            )
            predicted_frame = tf.layers.conv2d_transpose(
                hidden_8, 1, 1, 1, activation=tf.nn.sigmoid,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='predicted_frame_conv'
            )
            self._predicted_frame = tf.reshape(predicted_frame,
                                               [-1, 84, 84],
                                               name='predicted_frame_output')

            self._target_frame = tf.placeholder(
                'float32',
                shape=[None, convnet_pars['height'], convnet_pars['width']],
                name='target_frame')

            if self._predict_reward or self._predict_absorbing:
                hidden_9 = tf.layers.dense(
                    self._features,
                    128,
                    tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_9'
                )
                hidden_10 = tf.layers.dense(
                    hidden_9,
                    64,
                    tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_10'
                )
                if self._predict_reward:
                    self._target_reward = tf.placeholder(tf.int32,
                                                         shape=[None, 1],
                                                         name='target_reward')
                    self._target_reward_class = tf.clip_by_value(
                        self._target_reward, -1, 1,
                        name='target_reward_clipping') + 1
                    self._predicted_reward = tf.layers.dense(
                        hidden_10, 3, tf.nn.sigmoid, name='predicted_reward')
                    predicted_reward = tf.clip_by_value(
                        self._predicted_reward,
                        1e-7,
                        1 - 1e-7,
                        name='predicted_reward_clipping'
                    )
                    predicted_reward_logits = tf.log(
                        predicted_reward / (1 - predicted_reward),
                        name='predicted_reward_logits'
                    )
                    self._xent_reward = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.squeeze(self._target_reward_class),
                            logits=predicted_reward_logits,
                            name='sparse_softmax_cross_entropy_reward'
                        ),
                        name='xent_reward'
                    )
                if self._predict_absorbing:
                    self._target_absorbing = tf.placeholder(
                        tf.float32, shape=[None, 1], name='target_absorbing')
                    self._predicted_absorbing = tf.layers.dense(
                        hidden_10, 1, tf.nn.sigmoid, name='predicted_absorbing')
                    predicted_absorbing = tf.clip_by_value(
                        self._predicted_absorbing,
                        1e-7,
                        1 - 1e-7,
                        name='predicted_absorbing_clipping'
                    )
                    predicted_absorbing_logits = tf.log(
                        predicted_absorbing / (1 - predicted_absorbing),
                        name='predicted_absorbing_logits')
                    self._xent_absorbing = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.squeeze(self._target_absorbing),
                            logits=predicted_absorbing_logits,
                            name='sigmoid_cross_entropy_absorbing'
                        ),
                        name='xent_absorbing'
                    )

            predicted_frame = tf.clip_by_value(self._predicted_frame,
                                               1e-7,
                                               1 - 1e-7,
                                               name='predicted_frame_clipping')
            predicted_frame_logits = tf.log(
                predicted_frame / (1 - predicted_frame),
                name='predicted_frame_logits'
            )

            self._xent_frame = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self._target_frame,
                    logits=predicted_frame_logits,
                    name='sigmoid_cross_entropy_frame'
                ),
                name='xent_frame'
            )
            self._xent = self._xent_frame
            if self._predict_reward:
                self._xent += self._xent_reward
            if self._predict_absorbing:
                self._xent += self._xent_absorbing
            if self._contractive:
                raise NotImplementedError
            else:
                self._reg = tf.reduce_mean(tf.norm(self._features, 1, axis=1),
                                           name='reg')
            self._loss = tf.add(self._xent, self._reg_coeff * self._reg,
                                name='loss')

            tf.summary.scalar('xent_frame', self._xent_frame)
            if self._predict_reward:
                accuracy_reward = tf.equal(
                    tf.squeeze(self._target_reward_class),
                    tf.cast(tf.argmax(self._predicted_reward, 1), tf.int32)
                )
                self._accuracy_reward = tf.reduce_mean(
                    tf.cast(accuracy_reward, tf.float32))
                tf.summary.scalar('accuracy_reward', self._accuracy_reward)
                tf.summary.scalar('xent_reward', self._xent_reward)
            if self._predict_absorbing:
                accuracy_absorbing = tf.equal(
                    tf.squeeze(tf.cast(self._target_absorbing, tf.int32)),
                    tf.cast(tf.argmax(self._predicted_absorbing, 1), tf.int32)
                )
                self._accuracy_absorbing = tf.reduce_mean(
                    tf.cast(accuracy_absorbing, tf.float32))
                tf.summary.scalar('accuracy_absorbing',
                                  self._accuracy_absorbing)
                tf.summary.scalar('xent_absorbing', self._xent_absorbing)
            tf.summary.scalar('xent', self._xent)
            tf.summary.scalar('reg', self._reg)
            tf.summary.scalar('loss', self._loss)
            self._merged = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES,
                                  scope=self._scope_name)
            )

            optimizer = convnet_pars['optimizer']
            if optimizer['name'] == 'rmspropcentered':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer()
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer()
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_step = opt.minimize(loss=self._loss)

            initializer = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name) +
                tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                  scope=self._scope_name)
            )

        self._session.run(initializer)

        if self._folder_name is not None:
            self._train_writer = tf.summary.FileWriter(
                self._folder_name + '/' + self._scope_name,
                graph=tf.get_default_graph()
            )

        self._train_count = 0

        self._add_collection()

    @property
    def n_features(self):
        return self._n_features

    def _add_collection(self):
        tf.add_to_collection(self._scope_name + '_state', self._state)
        tf.add_to_collection(self._scope_name + '_action', self._action)
        tf.add_to_collection(self._scope_name + '_predicted_frame',
                             self._predicted_frame)
        tf.add_to_collection(self._scope_name + '_target_frame',
                             self._target_frame)
        if self._predict_reward:
            tf.add_to_collection(self._scope_name + '_predicted_reward',
                                 self._predicted_reward)
            tf.add_to_collection(self._scope_name + '_target_reward',
                                 self._target_reward)
            tf.add_to_collection(self._scope_name + '_target_reward_class',
                                 self._target_reward_class)
        if self._predict_absorbing:
            tf.add_to_collection(self._scope_name + '_predicted_absorbing',
                                 self._predicted_absorbing)
            tf.add_to_collection(self._scope_name + '_target_absorbing',
                                 self._target_absorbing)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self):
        self._state = tf.get_collection(self._scope_name + '_state')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]
        self._predicted_frame = tf.get_collection(
            self._scope_name + '_predicted_frame')[0]
        self._target_frame = tf.get_collection(
            self._scope_name + '_target_frame')[0]
        if self._predict_reward:
            self._predicted_reward = tf.get_collection(
                self._scope_name + '_predicted_reward')[0]
            tf.add_to_collection(self._scope_name + '_target_reward',
                                 self._target_reward)
            self._target_reward_class = tf.get_collection(
                self._scope_name + '_target_reward_class')[0]
        if self._predict_absorbing:
            self._predicted_absorbing = tf.get_collection(
                self._scope_name + '_predicted_absorbing')[0]
            self._target_absorbing = tf.get_collection(
                self._scope_name + '_target_absorbing')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]
