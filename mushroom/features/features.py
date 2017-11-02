from ._implementations.basis_features import BasisFeatures
from ._implementations.tiles_features import TilesFeatures
from ._implementations.tensorflow_features import TensorflowFeatures


def Features(basis_list=None, tilings=None, tensor_list=None, name=None,
             input_dim=None):
    if basis_list is not None and tilings is None and tensor_list is None:
        return BasisFeatures(basis_list)
    elif basis_list is None and tilings is not None and tensor_list is None:
        return TilesFeatures(tilings)
    elif basis_list is None and tilings is None and tensor_list is not None:
        return TensorflowFeatures(name, input_dim, tensor_list)
    else:
        raise ValueError('You must specify a list of basis or a list of tilings'
                         'or a list of tensors.')
