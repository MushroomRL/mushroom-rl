import torch
import torch.nn as nn

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.utils.features import uniform_grid
from mushroom_rl.utils.torch_utils import TorchUtils


class RadialBasisTensor(nn.Module):
    """
    Pytorch module to implement a basis function centered on a set of points. A subclass only implements
    ``_basis_function``, which is evaluated against every center at once: both the centers and the batch dimension
    are handled by broadcasting.

    """
    def __init__(self, mu, scale, dim=None, normalized=False):
        """
        Constructor.

        Args:
            mu (Array): centers of the basis functions;
            scale (Array): scales for the basis functions;
            dim (Array, None): list of dimension to be considered for the computation of the features. If None, all
                dimension are used to compute the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not.

        """
        super().__init__()

        self.register_buffer('_mu', TorchUtils.to_float_tensor(mu))
        self.register_buffer('_scale', TorchUtils.to_float_tensor(scale))
        if dim is not None:
            self.register_buffer('_dim', TorchUtils.to_int_tensor(dim))
        else:
            self._dim = None

        self._normalized = normalized

    def extra_repr(self):
        return f'n_centers={self.size}, normalized={self._normalized}'

    def forward(self, x):
        if self._dim is not None:
            x = torch.index_select(x, -1, self._dim)

        delta = x.unsqueeze(-2) - self._mu

        phi = self._basis_function(delta, self._scale)

        if self._normalized:
            return self._normalize(phi)
        else:
            return phi

    @classmethod
    def is_cyclic(cls):
        """
        Method used to change the basis generation in case of cyclic features.
        Returns:
            Whether the space we consider is cyclic or not.

        """
        return False

    @classmethod
    def generate(cls, n_centers, low, high, dimensions=None, eta=0.25, normalized=False):
        """
        Factory method that generates the list of dictionaries to build the tensors representing a set of uniformly
        spaced radial basis functions with `eta` overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be used for each dimension;
            low (Array): lowest value for each dimension;
            high (Array): highest value for each dimension;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. The number of
                dimensions must match the number of elements in ``high`` and ``low``;
            eta (float, 0.25): percentage of overlap between the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not.

        Returns:
            The tensor list.

        """
        low, high = ArrayBackend.convert(low, high, to='numpy')

        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)

        mu, w = uniform_grid(n_centers, low, high, eta, cls.is_cyclic())
        scale = cls._convert_to_scale(w)

        tensor_list = [cls(mu, scale, dimensions, normalized)]

        return tensor_list

    @property
    def size(self):
        return self._mu.shape[0]

    def _basis_function(self, delta, scale):
        raise NotImplementedError

    @staticmethod
    def _convert_to_scale(w):
        """
        Converts width of a basis function to scale

        Args:
            w (np.ndarray): array of widths of basis function for every dimension

        Returns:
            The array of scales for each basis function in any given dimension

        """
        raise NotImplementedError

    @staticmethod
    def _normalize(raw_phi):
        return torch.nan_to_num(raw_phi / torch.sum(raw_phi, -1, keepdim=True), 0.)


class GaussianRBFTensor(RadialBasisTensor):
    def _basis_function(self, delta, scale):
        return torch.exp(-torch.sum(delta ** 2 / scale, -1))

    @staticmethod
    def _convert_to_scale(w):
        return 2 * (w/3) ** 2


class VonMisesTensor(RadialBasisTensor):
    @classmethod
    def is_cyclic(cls):
        return True

    def _basis_function(self, delta, scale):
        return torch.exp(torch.sum(torch.cos(2*torch.pi*delta)/scale, -1) - torch.sum(1/scale))

    @staticmethod
    def _convert_to_scale(w):
        return w
