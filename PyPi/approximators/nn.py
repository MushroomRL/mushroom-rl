from keras.models import Sequential
from keras.layers.core import Dense
from keras.engine.topology import Layer

import tensorflow as tf


class NN(object):
    """
    Wrapper class of Keras's Sequential model.
    """
    def __init__(self, **approximator_params):
        """
        Constructor.

        # Arguments
            approximator_params (dict): parameters.
        """
        self.__name__ = 'DenseNN'

        self.approximator_params = approximator_params

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).
            y (np.array): target.
            fit_params (dict): other parameters.
        """
        if not hasattr(self, 'model'):
            self.model = self._init()

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        predictions = self.model.predict(x)

        return predictions.ravel()

    def __str__(self):
        return self.__name__


class DenseNN(NN):
    def _initialize(self):
        pars = self.approximator_params
        n_input = pars.pop('n_input')
        hidden_neurons = pars.pop('hidden_neurons')
        n_output = pars.pop('n_output')
        loss = pars.pop('loss')
        optimizer = pars.pop('optimizer')
        activation = pars.pop('activation', 'linear')
        metrics = pars.pop('metrics', None)

        model = Sequential()
        model.add(Dense(hidden_neurons[0], input_shape=(n_input,),
                        activation=activation, **pars))
        for i in range(1, len(hidden_neurons)):
            model.add(Dense(hidden_neurons[i], activation=activation, **pars))
        model.add(Dense(n_output, activation='linear', **pars))

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **pars)

        return model


class AtariConvNet(NN):
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

    def __initialize(self):
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)
        self.input_shape = input_shape
        self.target_size = target_size
        self.nb_actions = nb_actions
        self.encoding_dim = encoding_dim
        self.nb_epochs = nb_epochs
        self.dropout_prob = dropout_prob
        self.l1_alpha = l1_alpha
        self.binarize = binarize
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.logger = logger

        # Build network
        self.input = Input(shape=self.input_shape)
        self.u = Input(shape=(1,), dtype='int32')

        self.hidden = Convolution2D(32, 8, 8, border_mode='valid',
                                    activation='relu', subsample=(4, 4),
                                    dim_ordering='th')(self.input)
        self.hidden = Convolution2D(64, 4, 4, border_mode='valid',
                                    activation='relu', subsample=(2, 2),
                                    dim_ordering='th')(self.hidden)
        self.hidden = Convolution2D(64, 3, 3, border_mode='valid',
                                    activation='relu', subsample=(1, 1),
                                    dim_ordering='th')(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(self.encoding_dim, activation='relu')(self.hidden)
        self.output = Dense(self.target_size * self.nb_actions,
                            activation='linear',
                            activity_regularizer=l1(self.l1_alpha))(self.features)
        self.output_u = GatherLayer(self.target_size,
                                    self.nb_actions)([self.output, self.u])

        # Models
        self.model = Model(input=[self.input, self.u], output=self.output_u)
        self.encoder = Model(input=self.input, output=self.features)

        # Optimization algorithm
        self.optimizer = RMSProp()

        self.model.compile(optimizer=self.optimizer, loss='mse',
                           metrics=['accuracy'])
