import tensorflow as tf
from mushroom.features import tensors
from mushroom.utils.features import uniform_grid


def generate(n_centers, ranges):
    """
    Factory method that generates the list of dictionaries to build the tensors
    representing a set of uniformly spaced Gaussian radial basis functions with
    a 25\% overlap.

    Args:
        n_centers (list): list of the number of radial basis functions to be
            used for each dimension.
        ranges (list): list of two-elements lists specifying the range of
            each state variable.

    Returns:
        The list of dictionaries as described above.

    """
    n_features = len(ranges)
    assert len(n_centers) == n_features
    assert len(ranges[0]) == 2

    grid, b = uniform_grid(n_centers, ranges)

    tensor_list = list()
    for i in range(len(grid)):
        v = grid[i, :]
        bf = {'type': tensors.gaussian_tensor, 'params': [v, b]}
        tensor_list.append(bf)

    return tensor_list


def _generate(x, args):
    """
    Build the single tensor using the provided parameters.

    Args:
        x (tf.placeholder): the input placeholder;
        args (list): the parameters to build the single tensor.

    Returns:
        The tensor evaluating the features.

    """
    mu, scale = args

    v_list = list()
    for i, (m_i, s_i) in enumerate(zip(mu, scale)):
        v = (x[:, i] - m_i) ** 2 / s_i
        v_list.append(v)

    return tf.exp(-tf.add_n(v_list))
