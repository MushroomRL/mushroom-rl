import numpy as np

from ._implementations.basis_features import BasisFeatures
from ._implementations.functional_features import FunctionalFeatures
from ._implementations.tiles_features import TilesFeatures
from ._implementations.torch_features import TorchFeatures


def Features(basis_list=None, tilings=None, tensor_list=None,
             n_outputs=None, function=None):
    """
    Factory method to build the requested type of features. The types are
    mutually exclusive.

    Possible features are tilings (``tilings``), basis functions
    (``basis_list``), tensor basis (``tensor_list``), and functional mappings
    (``n_outputs`` and ``function``).

    The difference between ``basis_list`` and ``tensor_list`` is that the
    former is a list of python classes each one evaluating a single element of
    the feature vector, while the latter consists in a list  of PyTorch modules
    that can be used to build a PyTorch network. The use of ``tensor_list`` is
    a faster way to compute features than `basis_list` and is suggested when
    the computation of the requested features is slow (see the Gaussian radial
    basis function implementation as an example). A functional mapping applies
    a function to the input computing an ``n_outputs``-dimensional vector,
    where the mapping is expressed by ``function``. If ``function`` is not
    provided, the identity is used.

    Args:
        basis_list (list, None): list of basis functions;
        tilings ([object, list], None): single object or list of tilings;
        tensor_list (list, None): list of dictionaries containing the
            instructions to build the requested tensors;
        n_outputs (int, None): dimensionality of the feature mapping;
        function (object, None): a callable function to be used as feature
            mapping. Only needed when using a functional mapping.

    Returns:
        The class implementing the requested type of features.

    """
    if basis_list is not None and tilings is None and tensor_list is None and n_outputs is None:
        return BasisFeatures(basis_list)
    elif basis_list is None and tilings is not None and tensor_list is None and n_outputs is None:
        return TilesFeatures(tilings)
    elif basis_list is None and tilings is None and tensor_list is not None and n_outputs is None:
        return TorchFeatures(tensor_list)
    elif basis_list is None and tilings is None and tensor_list is None and n_outputs is not None:
        return FunctionalFeatures(n_outputs, function)
    else:
        raise ValueError('You must specify either: a list of basis, a list of tilings, '
                         'a list of tensors or the number of outputs '
                         '(and optionally the functionional mapping to use).')


def get_action_features(phi_state, action, n_actions):
    """
    Compute an array of size ``len(phi_state)`` * ``n_actions`` filled with
    zeros, except for elements from ``len(phi_state)`` * ``action`` to
    ``len(phi_state)`` * (``action`` + 1) that are filled with `phi_state`. This
    is used to compute state-action features.

    Args:
        phi_state (np.ndarray): the feature of the state;
        action (np.ndarray): the action whose features have to be computed;
        n_actions (int): the number of actions.

    Returns:
        The state-action features.

    """
    if len(phi_state.shape) > 1:
        assert phi_state.shape[0] == action.shape[0]

        phi = np.ones((phi_state.shape[0], n_actions * phi_state[0].size))
        i = 0
        for s, a in zip(phi_state, action):
            start = s.size * int(a[0])
            stop = start + s.size

            phi_sa = np.zeros(n_actions * s.size)
            phi_sa[start:stop] = s

            phi[i] = phi_sa

            i += 1
    else:
        start = phi_state.size * action[0]
        stop = start + phi_state.size

        phi = np.zeros(n_actions * phi_state.size)
        phi[start:stop] = phi_state

    return phi
