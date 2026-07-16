import numpy as np

from mushroom_rl.utils.features import uniform_grid

from .basis_function import BasisFunction


class GaussianRBF(BasisFunction):
    r"""
    Class implementing Gaussian radial basis functions. The value of the feature is computed using the formula:

    .. math::
        e^{-\sum \dfrac{(X_i - \mu_i)^2}{\sigma_i}}

    where :math:`X` is the input, :math:`\mu` is the mean vector and :math:`\sigma` is the scale parameter vector.

    """
    def __init__(self, mean, scale, dimensions=None):
        """
        Constructor.

        Args:
            mean (Array): the mean vector of the feature;
            scale (Array): the scale vector of the feature;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. The number of
                dimensions must match the dimensionality of ``mean`` and ``scale``.

        """
        self._mean, self._scale = self._to_numpy(mean, scale)

        super().__init__(dimensions)

        self._add_save_attr(_mean='numpy', _scale='numpy')

    def __repr__(self):
        name = f'GaussianRBF(mean={np.array2string(self._mean)}, scale={np.array2string(self._scale)}'
        if self._dim is not None:
            name += f', dimensions={self._dim}'

        return name + ')'

    @staticmethod
    def generate(n_centers, low, high, dimensions=None, eta=0.25):
        r"""
        Factory method to build uniformly spaced gaussian radial basis functions with `eta` overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be used for each selected dimension,
                in the same order they are declared in ``dimensions``;
            low (Array): lowest value for each dimension of the whole input;
            high (Array): highest value for each dimension of the whole input;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. If None,
                every dimension of the input is used;
            eta (float, 0.25): percentage of overlap between the features.

        Returns:
            The list of the generated radial basis functions.

        """
        low, high = BasisFunction._to_numpy(low, high)

        assert len(low) == len(high)

        if dimensions is not None:
            low, high = low[dimensions], high[dimensions]

        assert len(n_centers) == len(low)

        grid, w = uniform_grid(n_centers, low, high, eta)

        b = 2*(w/3)**2

        basis = list()
        for i in range(len(grid)):
            v = grid[i, :]
            bf = GaussianRBF(v, b, dimensions)
            basis.append(bf)

        return basis

    def _compute(self, x):
        return np.exp(-np.sum((x - self._mean)**2 / self._scale, axis=-1))
