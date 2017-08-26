import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense


class ConvNet:
    def __init__(self, n_actions, optimizer):
        # Build network
        input_layer = Input(shape=(84, 84, 4))

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
        self.output = Dense(n_actions, activation='linear')(features)

        # Models
        self.q = Model(outputs=[self.output], inputs=[input_layer])

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
        if isinstance(x, list):
            assert len(x) == 2

            actions = x[1].astype(np.int)

            out_all = self.q.predict(x[0], **fit_params)
            out = np.empty(out_all.shape[0])
            for i in xrange(out.size):
                out[i] = out_all[i, actions[i]]

            return out
        else:
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

    def set_weights(self, w):
        self.q.set_weights(w)

    def get_weights(self):
        return self.q.get_weights()
