import numpy as np


def uniform_grid(n_centers, low, high, eta=0.25, cyclic=False):
    """
    This function is used to create the parameters of uniformly spaced radial
    basis functions with `eta` of overlap. It creates a uniformly spaced grid of
    ``n_centers[i]`` points in each dimension i. Also returns a vector
    containing the appropriate width of the radial basis functions.

    Args:
         n_centers (list): number of centers of each dimension;
         low (np.ndarray): lowest value for each dimension;
         high (np.ndarray): highest value for each dimension;
         eta (float, 0.25): overlap between two radial basis functions;
         cyclic (bool, False): whether the state space is a ring or not

    Returns:
        The uniformly spaced grid and the width vector.

    """
    assert 0 < eta < 1.0

    n_features = len(low)
    w = np.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = low[i]
        end = high[i]
        # m = abs(start - end) / n
        if n == 1:
            w[i] = abs(end - start) / 2
            c_i = (start + end) / 2.
            c.append(np.array([c_i]))
        else:
            if cyclic:
                end_new = end - abs(end-start) / n
            else:
                end_new = end
            w[i] = (1 + eta) * abs(end_new - start) / n
            c_i = np.linspace(start, end_new, n)
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

    return grid, w
