import numpy as np
from mushroom.utils.features import uniform_grid


class GaussianRBF:
    """
    Class implementing Gaussian radial basis functions. The value of the feature
    is computed using the formula:
    
    .. math::
        \sum \dfrac{(X_i - \mu_i)^2}{\sigma_i}

    where X is the input, \mu is the mean vector and \sigma is the scale
    parameter vector.

    """
    def __init__(self, mean, scale, dimensions=None):
        """
        Constructor.

        Args:
            mean (np.ndarray): the mean vector of the feature;
            scale (np.ndarray): the scale vector of the feature;
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature. The number of dimensions must match
                the dimensionality of `mean` and `scale`.

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
    def generate(n_centers, ranges, dimensions=None):
        """
        Factory method to build uniformly spaced gaussian radial basis functions
        with a 25\% overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be
                used for each dimension.
            ranges (list): list of two-elements lists specifying the range of
                each state variable;
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature. The number of dimensions must match
                the number of elements in `n_centers` and `ranges`.

        Returns:
            The list of the generated radial basis functions.

        """
        n_features = len(ranges)
        assert len(n_centers) == n_features
        assert len(ranges[0]) == 2
        assert dimensions is None or n_features == len(dimensions)

        grid, b = uniform_grid(n_centers, ranges)

        basis = list()
        for i in xrange(len(grid)):
            v = grid[i, :]
            bf = GaussianRBF(v, b, dimensions)
            basis.append(bf)

        return basis
