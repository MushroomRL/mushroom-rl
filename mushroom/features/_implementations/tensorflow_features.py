import tensorflow as tf


class TensorflowFeatures:
    def __init__(self, name, input_dim, tensor_list):
        self._size = len(tensor_list)
        self._sess = tf.Session()

        with tf.variable_scope(name):
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
            self._phi = TensorflowFeatures.build_features(self._x, tensor_list)

    def __call__(self, x):
        if len(x.shape) == 1:
            x = [x]
        return self._sess.run(self._phi, feed_dict={self._x: x})

    @property
    def size(self):
        return self._size

    @staticmethod
    def build_features(x, tensor_list):
        basis_functions = []

        for tensor in tensor_list:
            tensor_type = tensor['type']
            parameters = tensor['params']
            bf = tensor_type._generate(x, parameters)
            basis_functions.append(bf)

        return tf.concat(basis_functions, axis=0)