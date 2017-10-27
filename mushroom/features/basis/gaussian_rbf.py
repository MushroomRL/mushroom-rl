import numpy as np


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
    def uniform_grid(tot_points, n_features, c):

        n_rows = 1
        n_cols = 0

        grid = np.zeros((tot_points, n_features))

        for discrete_values in c:
            i1 = 0
            dim = len(discrete_values)

            for i in xrange(dim):
                for r in xrange(n_rows):
                    idxr = r + i * n_rows
                    for c in xrange(n_cols):
                        grid[idxr, c] = grid[r, c]
                    grid[idxr, n_cols] = discrete_values[i1]

                i1 += 1

            n_cols += 1
            n_rows *= len(discrete_values)

        return grid

    @staticmethod
    def generate(n_centers, ranges, dimensions=None):
        n_features = len(ranges)
        assert len(n_centers) == n_features
        assert len(ranges[0]) == 2
        assert dimensions is None or n_features == len(dimensions)

        basis = list()
        b = np.zeros(n_features)
        c = list()
        totpoints = 1
        for i, n in enumerate(n_centers):
            start = ranges[i][0]
            end = ranges[i][1]

            b[i] = (end - start)**2 / n**3
            m = abs(start - end) / n
            if n == 1:
                c_i = (start + end) / 2.
                c.append(np.array([c_i]))
            else:
                c_i = np.linspace(start - m * .1, end + m * .1, n)
                c.append(c_i)
            totpoints *= n

        grid = GaussianRBF.uniform_grid(totpoints, n_features, c)

        for i in xrange(totpoints):
            v = grid[i, :]
            bf = GaussianRBF(v, b)
            basis.append(bf)

        return basis
