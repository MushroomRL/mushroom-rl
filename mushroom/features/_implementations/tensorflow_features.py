import numpy as np
import tensorflow as tf

from .features_implementation import FeaturesImplementation


class TensorflowFeatures(FeaturesImplementation):
    def __init__(self, name, input_dim, tensor_list):
        self._size = len(tensor_list)
        self._sess = tf.Session()

        with tf.variable_scope(name):
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],
                                     name='x')
            self._phi = TensorflowFeatures.build_features(self._x, tensor_list)

    def __call__(self, *args):
        if len(args) > 1:
            x = np.concatenate(args, axis=-1)
        else:
            x = args[0]

        x = np.atleast_2d(x)

        y = self._sess.run(self._phi, feed_dict={self._x: x})

        if len(y) == 1:
            y = y[0]
        else:
            y = np.array(y)

        return y

    @property
    def size(self):
        return self._size

    @staticmethod
    def build_features(x, tensor_list):
        basis_functions = list()

        for tensor in tensor_list:
            tensor_type = tensor['type']
            parameters = tensor['params']
            bf = tensor_type._generate(x, parameters)
            basis_functions.append(bf)

        return tf.stack(basis_functions, axis=-1)
