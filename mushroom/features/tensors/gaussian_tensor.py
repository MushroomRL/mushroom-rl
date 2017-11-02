import tensorflow as tf
from mushroom.features import tensors
from mushroom.utils.features import uniform_grid


def generate(n_centers, ranges):
    n_features = len(ranges)
    assert len(n_centers) == n_features
    assert len(ranges[0]) == 2

    grid, b = uniform_grid(n_centers, ranges)

    tensor_list = list()
    for i in xrange(len(grid)):
        v = grid[i, :]
        bf = {'type': tensors.gaussian_tensor, 'params': [v, b]}
        tensor_list.append(bf)

    return tensor_list


def _generate(x, args):
    mu, scale = args

    v_list = []
    for i, (m_i, s_i) in enumerate(zip(mu, scale)):
        v = (x[:, i] - m_i) ** 2 / s_i
        v_list.append(v)

    return tf.exp(-tf.add_n(v_list))
