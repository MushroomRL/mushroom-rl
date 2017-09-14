import tensorflow as tf


class Extractor:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name
        self._reg_coeff = convnet_pars.get('reg_coeff', 0.)

        self._session = tf.Session()

        if load_path is not None:
            self._load(load_path)
        else:
            self._build(convnet_pars)

        self._train_saver = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=self._scope_name))

    def predict(self, x, reconstruction=False):
        if not reconstruction:
            return self._session.run(self._features, feed_dict={self._x: x})
        else:
            return self._session.run(self._prediction, feed_dict={self._x: x})

    def get_loss(self, y_t, y, x):
        return self._session.run(self._loss,
                                 feed_dict={self._target_prediction: y_t,
                                            self._prediction: y,
                                            self._x: x})

    def train_on_batch(self, x, y):
        summaries, _, self.loss = self._session.run(
            [self._merged, self._train_step, self._loss],
            feed_dict={self._x: x, self._target_prediction: y}
        )
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
            self._x = tf.placeholder(tf.float32,
                                     shape=[None,
                                            convnet_pars['height'],
                                            convnet_pars['width'],
                                            convnet_pars['history_length']],
                                     name='input')
            hidden_1 = tf.layers.conv2d(
                self._x, 32, 8, 4, activation=tf.nn.relu,
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
            self._features = tf.reshape(hidden_4, [-1, 5 * 5 * 16],
                                        name='features')
            hidden_5 = tf.layers.conv2d_transpose(
                hidden_4, 16, 3, 1, activation=tf.nn.relu,
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
            self._prediction = tf.layers.conv2d_transpose(
                hidden_8, 4, 1, 1, activation=tf.nn.sigmoid,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='prediction'
            )

            self._target_prediction = tf.placeholder(
                'float32',
                shape=[None, convnet_pars['height'], convnet_pars['width'],
                       convnet_pars['history_length']],
                name='target_prediction')

            prediction = tf.clip_by_value(self._prediction, 1e-7, 1 - 1e-7)
            prediction_logits = tf.log(prediction / (1 - prediction))
            self._xent = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self._target_prediction,
                    logits=prediction_logits
                )
            )
            self._reg = tf.reduce_mean(tf.norm(self._features, 1, axis=1))
            self._loss = self._xent + self._reg_coeff * self._reg
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
                                  scope=self._scope_name))

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
        return self._features.shape[1]

    def _add_collection(self):
        tf.add_to_collection(self._scope_name + '_x', self._x)
        tf.add_to_collection(self._scope_name + '_prediction', self._prediction)
        tf.add_to_collection(self._scope_name + '_target_prediction',
                             self._target_prediction)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._prediction = tf.get_collection(
            self._scope_name + '_prediction')[0]
        self._target_prediction = tf.get_collection(
            self._scope_name + '_target_prediction')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]
