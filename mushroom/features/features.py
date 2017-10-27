from ._implementations.basis_features import BasisFeatures
from ._implementations.tiles_features import TilesFeatures


def Features(basis_list=None, tilings=None):
    if basis_list is not None:
        return BasisFeatures(basis_list)
    elif tilings is not None:
        return TilesFeatures(tilings)
    else:
        raise ValueError('You must specify a set of basis or a set of tilings')
