import numpy as np
from mushroom.distributions import *
from mushroom.utils.numerical_gradient import numerical_diff_dist

def gaussian():
    print('Testing gaussian distribution...')
    n_dims = 6

    random_matrix = np.random.rand(n_dims, n_dims)
    sigma = random_matrix.dot(random_matrix.T)

    mu = np.random.randn(n_dims)

    dist = GaussianDistribution(mu, sigma)

    for i in range(20):
        theta = dist.sample()
        exact_diff = dist.diff(theta)
        numerical_diff = numerical_diff_dist(dist, theta)

        assert np.allclose(exact_diff, numerical_diff)


def cholesky_gaussian():
    print('Testing gaussian cholesky distribution...')
    n_dims = 6

    random_matrix = np.random.rand(n_dims, n_dims)
    sigma = random_matrix.dot(random_matrix.T)

    mu = np.random.randn(n_dims)

    dist = GaussianCholeskyDistribution(mu, sigma)

    for i in range(20):
        theta = dist.sample()
        exact_diff = dist.diff(theta)
        numerical_diff = numerical_diff_dist(dist, theta)

        assert np.allclose(exact_diff, numerical_diff)


if __name__ == '__main__':
    print('Executing policy test...')

    gaussian()
    cholesky_gaussian()
