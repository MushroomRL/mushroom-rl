import numpy as np


def uniform_grid(n_centers, low, high):
    """
    This function is used to create the parameters of uniformly spaced radial
    basis functions with 25% of overlap. It creates a uniformly spaced grid of
    ``n_centers[i]`` points in each ``ranges[i]``. Also returns a vector
    containing the appropriate scales of the radial basis functions.

    Args:
         n_centers (list): number of centers of each dimension;
         low (np.ndarray): lowest value for each dimension;
         high (np.ndarray): highest value for each dimension.

    Returns:
        The uniformly spaced grid and the scale vector.

    """
    n_features = len(low)
    b = np.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = low[i]
        end = high[i]

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

        for i in range(dim):
            for r in range(n_rows):
                idx_r = r + i * n_rows
                for c in range(n_cols):
                    grid[idx_r, c] = grid[r, c]
                grid[idx_r, n_cols] = discrete_values[i1]

            i1 += 1

        n_cols += 1
        n_rows *= len(discrete_values)

    return grid, b
