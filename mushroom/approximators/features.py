from .implementations.features.tiles_features import TilesFeatures
from .implementations.features.basis_features import BasisFeatures


def Features(basis_list=None, tilings=None):
    if basis_list is not None:
        return BasisFeatures(basis_list)
    elif tilings is not None:
        return TilesFeatures(tilings)
    else:
        raise NotImplementedError()
