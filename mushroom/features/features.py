from ._implementations.basis_features import BasisFeatures
from ._implementations.tiles_features import TilesFeatures
from ._implementations.tensorflow_features import TensorflowFeatures


def Features(basis_list=None, tilings=None, tensor_list=None, name=None,
             input_dim=None):
    """
    Factory method to build the requested type of features. The types are
    mutually exclusive.

    The difference between `basis_list` and `tensor_list` is that the former
    is a list of python classes each one evaluating a single element of the
    feature vector, while the latter consists in a dictionary that can be used
    to build a Tensorflow graph. The use of `tensor_list` is a faster way to
    compute features than `basis_list` and is suggested when the computation
    of the requested features is slow (see the Gaussian radial basis function
    implementation as an example).

    Args:
        basis_list (list, None): list of basis functions;
        tilings ([object, list], None): single object or list of tilings;
        tensor_list (list, None): list of dictionaries containing the
            instructions to build the requested tensors;
        name (str, None): name of the group of tensors. Only needed when
            using a list of tensors;
        input_dim (int, None): the dimension of the input state. Only needed
            when using a list of tensors.

    Returns:
        The class implementing the requested type of features.

    """
    if basis_list is not None and tilings is None and tensor_list is None:
        return BasisFeatures(basis_list)
    elif basis_list is None and tilings is not None and tensor_list is None:
        return TilesFeatures(tilings)
    elif basis_list is None and tilings is None and tensor_list is not None:
        return TensorflowFeatures(name, input_dim, tensor_list)
    else:
        raise ValueError('You must specify a list of basis or a list of tilings'
                         'or a list of tensors.')
