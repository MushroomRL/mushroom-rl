import numpy as np
from mushroom_rl.utils.features import uniform_grid


class GaussianRBF:
    r"""
    Class implementing Gaussian radial basis functions. The value of the feature is computed using the formula:
    
    .. math::
        \sum \dfrac{(X_i - \mu_i)^2}{\sigma_i}

    where :math:`X` is the input, :math:`\mu` is the mean vector and :math:`\sigma` is the scale parameter vector.

    """
    def __init__(self, mean, scale, dimensions=None):
        """
        Constructor.

        Args:
            mean (np.ndarray): the mean vector of the feature;
            scale (np.ndarray): the scale vector of the feature;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. The number of
                dimensions must match the dimensionality of ``mean`` and ``scale``.

        """
        self._mean = mean
        self._scale = scale
        self._dim = dimensions

    def __call__(self, x):
        if self._dim is not None:
            x = x[self._dim]

        return np.exp(-np.sum((x - self._mean)**2 / self._scale))

    def __str__(self):
        name = 'GaussianRBF ' + str(self._mean) + ' ' + str(self._scale)
        if self._dim is not None:
            name += ' ' + str(self._dim)
        return name

    @staticmethod
    def generate(n_centers, low, high, dimensions=None, eta=0.25):
        r"""
        Factory method to build uniformly spaced gaussian radial basis functions with `eta` overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be used for each dimension.
            low (np.ndarray): lowest value for each dimension;
            high (np.ndarray): highest value for each dimension;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. The number of
                dimensions must match the number of elements in ``n_centers`` and ``low``;
            eta (float, 0.25): percentage of overlap between the features.

        Returns:
            The list of the generated radial basis functions.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)

        grid, w = uniform_grid(n_centers, low, high, eta)

        b = 2*(w/3)**2

        basis = list()
        for i in range(len(grid)):
            v = grid[i, :]
            bf = GaussianRBF(v, b, dimensions)
            basis.append(bf)

        return basis
