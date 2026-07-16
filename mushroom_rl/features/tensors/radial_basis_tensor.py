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
    def __init__(self, mu, scale, dimensions=None, normalized=False):
        """
        Constructor.

        Args:
            mu (Array): centers of the basis functions;
            scale (Array): scales for the basis functions;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. If None, all
                dimensions are used to compute the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not.

        """
        super().__init__()

        self.register_buffer('_mu', TorchUtils.to_float_tensor(mu))
        self.register_buffer('_scale', TorchUtils.to_float_tensor(scale))
        if dimensions is not None:
            self.register_buffer('_dim', TorchUtils.to_int_tensor(dimensions))
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
            n_centers (list): list of the number of radial basis functions to be used for each selected dimension,
                in the same order they are declared in ``dimensions``;
            low (Array): lowest value for each dimension of the whole input;
            high (Array): highest value for each dimension of the whole input;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. If None,
                every dimension of the input is used;
            eta (float, 0.25): percentage of overlap between the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not.

        Returns:
            The tensor list.

        """
        low, high = ArrayBackend.convert(low, high, to='numpy')

        assert len(low) == len(high)

        if dimensions is not None:
            low, high = low[dimensions], high[dimensions]

        assert len(n_centers) == len(low)

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
        Converts the width of a basis function to its scale.

        Args:
            w (Array): array of widths of the basis function for every dimension.

        Returns:
            The array of scales for each basis function in any given dimension.

        """
        raise NotImplementedError

    @staticmethod
    def _normalize(raw_phi):
        return torch.nan_to_num(raw_phi / torch.sum(raw_phi, -1, keepdim=True), 0.)


class GaussianRBFTensor(RadialBasisTensor):
    r"""
    Pytorch module implementing Gaussian radial basis functions. The value of the feature is computed using the
    formula:

    .. math::
        e^{-\sum \dfrac{(X_i - \mu_i)^2}{\sigma_i}}

    where :math:`X` is the input, :math:`\mu` is the mean vector and :math:`\sigma` is the scale parameter vector.
    This is the tensor counterpart of ``GaussianRBF``.

    """
    def _basis_function(self, delta, scale):
        return torch.exp(-torch.sum(delta ** 2 / scale, -1))

    @staticmethod
    def _convert_to_scale(w):
        return 2 * (w/3) ** 2


class VonMisesTensor(RadialBasisTensor):
    r"""
    Pytorch module implementing Von Mises basis functions, i.e. the cyclic counterpart of the Gaussian radial basis
    functions, to be used on angular inputs. The value of the feature is computed using the formula:

    .. math::
        e^{\sum \dfrac{\cos{2\pi(X_i - \mu_i)}}{\sigma_i} - \sum \dfrac{1}{\sigma_i}}

    where :math:`X` is the input, :math:`\mu` is the mean vector and :math:`\sigma` is the scale parameter vector.
    The second sum normalizes the feature to one in its center.

    """
    @classmethod
    def is_cyclic(cls):
        return True

    def _basis_function(self, delta, scale):
        return torch.exp(torch.sum(torch.cos(2*torch.pi*delta)/scale, -1) - torch.sum(1/scale))

    @staticmethod
    def _convert_to_scale(w):
        return w
