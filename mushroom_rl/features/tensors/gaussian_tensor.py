import torch
import torch.nn as nn
from mushroom_rl.utils.features import uniform_grid


class PyTorchGaussianRBF(nn.Module):
    """
    Pytorch module to implement a gaussian radial basis function.

    """
    def __init__(self, mu, scale, dim):
        self._mu = torch.from_numpy(mu)
        self._scale = torch.from_numpy(scale)
        if dim is not None:
            self._dim = torch.from_numpy(dim)
        else:
            self._dim = None

    def forward(self, x):
        if self._dim is not None:
            x = torch.index_select(x, 1, self._dim)

        delta = x - self._mu

        return torch.exp(-torch.sum(delta**2 / self._scale, 1))

    @staticmethod
    def generate(n_centers, low, high,  dimensions=None):
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
                the number of elements in ``n_centers`` and ``low``.

        Returns:
            The list of dictionaries as described above.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)

        grid, scale = uniform_grid(n_centers, low, high)

        tensor_list = list()
        for i in range(len(grid)):
            mu = grid[i, :]
            tensor_list.append(PyTorchGaussianRBF(mu, scale, dimensions))

        return tensor_list


