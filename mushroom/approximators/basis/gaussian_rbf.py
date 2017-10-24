import numpy as np


class GaussianRBF:
    def __init__(self, mean, scale, dimensions=None):
        self._mean = mean
        self._scale = scale
        self._dim = dimensions

    def __call__(self, x):

        if self._dim is not None:
            x = x[self._dim]

        v = 0.0
        for x_i, m_i, s_i in zip(x, self._mean, self._scale):
            v += (x_i - m_i)**2 / s_i
        return np.exp(-v)

    def __str__(self):
        name = 'GaussianRBF '+ str(self._mean) + ' ' + str(self._scale)
        if self._dim is not None:
            name += ' ' + str(self._dim)
        return name

    @staticmethod
    def uniform_grid(grid, n_rows, n_cols, discrete_values):
        i1 = 0
        dim = len(discrete_values)

        for i in xrange(dim):
            for r in xrange(n_rows):
                idxr = r + i * n_rows
                for c in xrange(n_cols):
                    grid[idxr, c] = grid(r, c)
                grid[idxr, n_cols] = discrete_values[i1]

            i1 += 1

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
                c_i = (start + end)/2.0
                c.append(np.array([c_i]))
            else:
                c_i = np.linspace(start - m * 0.1, end + m * 0.1, n)
                c.append(c_i)
            totpoints *= n


        grid_nrows = 1
        grid_ncols = 0

        grid = np.zeros((totpoints, n_features))

        for c_i in c:
            GaussianRBF.uniform_grid(grid, grid_nrows, grid_ncols, c_i)
            grid_nrows += 1
            grid_ncols *= len(c_i)

        for i in xrange(totpoints):
            v = grid[i, :]
            bf = GaussianRBF(v, b)
            basis.append(bf)


        return basis



from mushroom.approximators.features import Features
basis_list = GaussianRBF.generate([3, 3], [ [0.0, 1.0], [0.0, 1.0]])

phi=Features(basis_list=basis_list)

input1 = np.array([0.1, 0.1])
input2 = np.array([0.5, 0.5])
input3 = np.array([0.1, 0.9])
input4 = np.array([0.9, 0.1])

print 'phi(1)'
print phi(input1)
print 'phi(2)'
print phi(input2)
print 'phi(3)'
print phi(input3)
print 'phi(4)'
print phi(input4)