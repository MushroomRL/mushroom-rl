import numpy as np
from mushroom.utils.features import uniform_grid


class GaussianRBF:
    def __init__(self, mean, scale, dimensions=None):
        self._mean = mean
        self._scale = scale
        self._dim = dimensions

    def __call__(self, x):

        if self._dim is not None:
            x = x[self._dim]

        v = 0.
        for x_i, m_i, s_i in zip(x, self._mean, self._scale):
            v += (x_i - m_i)**2 / s_i
        return np.exp(-v)

    def __str__(self):
        name = 'GaussianRBF ' + str(self._mean) + ' ' + str(self._scale)
        if self._dim is not None:
            name += ' ' + str(self._dim)
        return name

    @staticmethod
    def generate(n_centers, ranges, dimensions=None):
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