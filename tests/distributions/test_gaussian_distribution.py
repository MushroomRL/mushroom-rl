import numpy as np
from mushroom_rl.distributions import *
from mushroom_rl.utils.numerical_gradient import numerical_diff_dist


def test_gaussian():
    np.random.seed(88)
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

    theta = dist.sample()
    log_p = dist.log_pdf(theta)
    log_p_test = -5.622792079250801
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)
    assert np.array_equal(dist.get_parameters(), theta.mean(axis=0))

    dist.mle(theta, weights)
    assert np.array_equal(dist.get_parameters(), weights.dot(theta) / np.sum(weights))

    entropy = dist.entropy()
    assert np.isclose(entropy, 4.74920830903762)


def test_diagonal_gaussian():
    np.random.seed(88)
    n_dims = 6

    std = np.abs(np.random.rand(n_dims))
    mu = np.random.randn(n_dims)

    dist = GaussianDiagonalDistribution(mu, std)

    for i in range(20):
        theta = dist.sample()
        exact_diff = dist.diff(theta)
        numerical_diff = numerical_diff_dist(dist, theta)

        assert np.allclose(exact_diff, numerical_diff)

    theta = dist.sample()
    log_p = dist.log_pdf(theta)
    log_p_test = -7.818947754486631
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)
    assert np.array_equal(dist.get_parameters()[:n_dims], theta.mean(axis=0))
    assert np.array_equal(dist.get_parameters()[n_dims:], theta.std(axis=0))

    dist.mle(theta, weights)
    wmle_test = np.array([0.14420612, -0.02660736,  0.07439633, -0.00424596,
                          0.2495252 , 0.20968329,  0.97527594,  1.08187144,
                          1.04907019,  1.0634484 , 1.03453275,  1.03961039])
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.749505679983452)


def test_cholesky_gaussian():
    np.random.seed(88)
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

    theta = dist.sample()
    log_p = dist.log_pdf(theta)
    log_p_test = -5.622792079250836
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)

    mle_test = np.array([0.03898376, 0.07252868,  0.26070238,  0.13782173,  0.18927999, -0.02548812,
                         1.00855691, 0.19697324,  1.06381216, -0.07439629,  0.0656251 ,  1.02907183,
                         0.03779866,-0.00504703, -0.14902275,  0.99917335, -0.09132656, -0.03225904,
                        -0.13589437, 0.1419549,  0.94040997, -0.00145945,  0.00326904,  0.00291136,
                        -0.07456335, 0.04581934,  1.02750578])
    assert np.allclose(dist.get_parameters(), mle_test)

    dist.mle(theta, weights)
    wmle_test = np.array([-0.07797052, 0.08518447,  0.36272218, 0.17409145,  0.26339453, -0.02891896,
                           0.98529941, 0.26728657,  1.09177517, -0.03838698, -0.08395759,  0.98168805,
                           0.0150622, 0.05611417, -0.09351055,  1.0166716, -0.06390746, -0.05409177,
                          -0.08944413, 0.17745539,  1.01277413, 0.00923361,  0.05694206, -0.02457328,
                          -0.14141649, 0.1117947,  1.03121418])
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.628109062120682)
