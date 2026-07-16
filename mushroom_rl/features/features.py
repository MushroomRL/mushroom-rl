import torch.nn as nn

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.features.basis import BasisFunction
from mushroom_rl.features.tiles import AbstractTiles


class Features(MushroomObject):
    """
    Class building the requested type of features. The type is inferred from the elements of ``feature_list``, which
    can be a single object or a list of tilings, basis functions or tensor basis. Different types of features cannot
    be mixed in the same ``feature_list``. Alternatively, a functional mapping is built by specifying ``n_outputs``
    (and optionally ``function``).

    The difference between basis functions and tensor basis is that the former is a list of python classes each one
    evaluating a single element of the feature vector, while the latter consists in a list of PyTorch modules that can
    be used to build a PyTorch network, e.g. with ``to_torch_module``. The use of the tensor basis is a faster way to
    compute features than the basis functions and is suggested when the computation of the requested features is slow
    (see the Gaussian radial basis function implementation as an example). A functional mapping applies a function to
    the input computing an ``n_outputs``-dimensional vector, where the mapping is expressed by ``function``. If
    ``function`` is not provided, the identity is used.

    Args:
        feature_list ([object, list], None): single object or list of tilings, basis functions or tensor basis;
        n_outputs (int, None): dimensionality of the feature mapping;
        function (object, None): a callable function to be used as feature mapping. Only needed when using a
            functional mapping. The raw input is passed to it as it is, so it must handle both a single 1-D
            input and a 2-D batch;
        backend (str, 'numpy'): the backend of the computed features, i.e. ``'numpy'`` or ``'torch'``.

    """
    def __new__(cls, feature_list=None, n_outputs=None, function=None, backend='numpy'):
        if cls is not Features:
            return MushroomObject.__new__(cls)

        if feature_list is not None:
            if n_outputs is not None or function is not None:
                raise ValueError('The feature_list and the functional mapping arguments (n_outputs, function) are '
                                 'mutually exclusive.')

            feature_list = Features._as_list(feature_list)

            if len(feature_list) == 0:
                raise ValueError('The feature_list must not be empty.')
            elif all(isinstance(f, nn.Module) for f in feature_list):
                impl = TorchFeatures
            elif all(isinstance(f, AbstractTiles) for f in feature_list):
                impl = TilesFeatures
            elif all(isinstance(f, BasisFunction) for f in feature_list):
                impl = BasisFeatures
            else:
                raise ValueError('The feature_list must be a homogeneous list of basis functions, tilings or '
                                 'basis tensors.')

            return MushroomObject.__new__(impl)
        elif n_outputs is not None:
            return MushroomObject.__new__(FunctionalFeatures)
        else:
            raise ValueError('You must specify either a feature_list (basis functions, tilings or basis tensors) or '
                             'the number of outputs (and optionally the functional mapping to use).')

    def __init__(self, backend='numpy', internal_backend='numpy'):
        """
        Constructor.

        Args:
            backend (str, 'numpy'): the backend of the computed features, i.e. ``'numpy'`` or ``'torch'``;
            internal_backend (str, 'numpy'): the backend the features are computed in, set by the implementation.
                The raw input is converted into it before computing the features.

        """
        self._backend = backend
        self._internal_backend = internal_backend

        self._add_save_attr(_backend='primitive', _internal_backend='primitive')

    def __call__(self, *x):
        """
        Evaluate the feature vector in the given raw input. If more than one element is passed, the raw input is
        concatenated before computing the features. The raw input is converted into the internal backend before the
        computation, and the features are converted into the backend selected at construction after it.

        Args:
            *x: the raw input.

        Returns:
            The features vector computed from the raw input.

        """
        y = self._compute(self._convert_input(x))

        return ArrayBackend.convert(y, to=self._backend)

    def to_torch_module(self):
        """
        Returns:
            A differentiable, gradient-enabled torch module computing the features. Only supported by tensor features.

        """
        raise NotImplementedError('Only torch features can be converted to a differentiable module.')

    @staticmethod
    def get_action_features(phi_state, action, n_actions):
        """
        Compute an array of size ``len(phi_state)`` * ``n_actions`` filled with
        zeros, except for elements from ``len(phi_state)`` * ``action`` to
        ``len(phi_state)`` * (``action`` + 1) that are filled with `phi_state`. This
        is used to compute state-action features.

        Both a single feature vector and a batch of them are supported. A batch of a single sample is collapsed
        into a single feature vector, consistently with the features computation. The state-action features are
        computed in the backend of ``phi_state``.

        Args:
            phi_state (Array): the feature of the state, or a batch of them;
            action (Array): the action whose features have to be computed, or a batch of them;
            n_actions (int): the number of actions.

        Returns:
            The state-action features.

        """
        if action.shape[0] == 1:
            phi_state = phi_state.reshape(-1)
            action = action.reshape(-1)

        backend = ArrayBackend.get_array_backend_from(phi_state)
        n_features = phi_state.shape[-1]

        if len(phi_state.shape) > 1:
            assert phi_state.shape[0] == action.shape[0]

            n_samples = phi_state.shape[0]

            phi = backend.zeros(n_samples, n_actions * n_features)
            rows = backend.arange(0, n_samples).reshape(-1, 1)
            columns = action.reshape(-1, 1) * n_features + backend.arange(0, n_features).reshape(1, -1)

            phi[rows, columns] = phi_state
        else:
            start = int(action[0]) * n_features

            phi = backend.zeros(n_actions * n_features)
            phi[start:start + n_features] = phi_state

        return phi

    @property
    def size(self):
        """
        Returns:
             The number of elements in the features vector.

        """
        raise NotImplementedError

    def _compute(self, x):
        """
        Compute the features vector from the concatenated raw input. The input is already converted into the
        internal backend, and the conversion of the output into the requested backend is done by ``__call__``.

        Args:
            x: the concatenated raw input, in the internal backend.

        Returns:
            The features vector computed from the raw input, in the internal backend.

        """
        raise NotImplementedError

    def _convert_input(self, args):
        args = [ArrayBackend.convert(x, to=self._internal_backend) for x in args]

        if len(args) > 1:
            return ArrayBackend.get_array_backend(self._internal_backend).concatenate(args, dim=-1)

        return args[0]

    @staticmethod
    def _as_list(feature_list):
        return feature_list if isinstance(feature_list, list) else [feature_list]


from ._impl import BasisFeatures, TilesFeatures, TorchFeatures, FunctionalFeatures  # noqa: E402
