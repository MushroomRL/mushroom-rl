import numpy as np


def uniform_grid(n_centers, ranges):
    n_features = len(ranges)
    b = np.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = ranges[i][0]
        end = ranges[i][1]

        b[i] = (end - start) ** 2 / n ** 3
        m = abs(start - end) / n
        if n == 1:
            c_i = (start + end) / 2.
            c.append(np.array([c_i]))
        else:
            c_i = np.linspace(start - m * .1, end + m * .1, n)
            c.append(c_i)
        tot_points *= n

    n_rows = 1
    n_cols = 0

    grid = np.zeros((tot_points, n_features))

    for discrete_values in c:
        i1 = 0
        dim = len(discrete_values)

        for i in xrange(dim):
            for r in xrange(n_rows):
                idx_r = r + i * n_rows
                for c in xrange(n_cols):
                    grid[idx_r, c] = grid[r, c]
                grid[idx_r, n_cols] = discrete_values[i1]

            i1 += 1

        n_cols += 1
        n_rows *= len(discrete_values)

    return grid, b
