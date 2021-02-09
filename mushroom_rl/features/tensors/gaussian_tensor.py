import torch
import torch.nn as nn
from mushroom_rl.utils.features import uniform_grid
from mushroom_rl.utils.torch import to_float_tensor, to_int_tensor


class GaussianRBFTensor(nn.Module):
    """
    Pytorch module to implement a gaussian radial basis function.

    """
    def __init__(self, mu, scale, dim, use_cuda):
        """
        Constructor.

        Args:
            mu (np.ndarray): centers of the gaussian RBFs;
            scale (np.ndarray): scales for the RBFs;
            dim (np.ndarray): list of dimension to be considered for the computation of the features;
            use_cuda (bool): whether to use cuda for the computation or not.

        """
        self._mu = to_float_tensor(mu, use_cuda)
        self._scale = to_float_tensor(scale, use_cuda)
        if dim is not None:
            self._dim = to_int_tensor(dim, use_cuda)
        else:
            self._dim = None

        self._use_cuda = use_cuda

    def forward(self, x):
        if self._use_cuda:
            x = x.cuda()
        if self._dim is not None:
            x = torch.index_select(x, 1, self._dim)

        x = x.unsqueeze(1).repeat(1, self._mu.shape[0], 1)
        delta = x - self._mu.repeat(x.shape[0], 1, 1)
        return torch.exp(-torch.sum(delta**2 / self._scale, -1)).squeeze(-1)

    @staticmethod
    def generate(n_centers, low, high,  dimensions=None, use_cuda=False):
        """
        Factory method that generates the list of dictionaries to build the
        tensors representing a set of uniformly spaced Gaussian radial basis
        functions with a 25% overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be
                              used for each dimension;
            low (np.ndarray): lowest value for each dimension;
            high (np.ndarray): highest value for each dimension;
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature. The number of dimensions must match
                the number of elements in ``n_centers`` and ``low``;
            use_cuda (bool): whether to use cuda for the computation or not.

        Returns:
            The list of dictionaries as described above.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)

        mu, scale = uniform_grid(n_centers, low, high)

        tensor_list = [GaussianRBFTensor(mu, scale, dimensions, use_cuda)]

        return tensor_list

    @property
    def size(self):
        return self._mu.shape[0]
