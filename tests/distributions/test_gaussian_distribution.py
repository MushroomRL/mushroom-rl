import numpy as np

from helper.dummies import FakeLogger

from mushroom_rl.distributions import GaussianDistribution, GaussianDiagonalDistribution, GaussianCholeskyDistribution
from mushroom_rl.utils.numerical_gradient import numerical_diff_dist


def test_gaussian():
    np.random.seed(42)
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
    log_p_test = -6.475847829950508
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)
    assert np.array_equal(dist.get_parameters(), theta.mean(axis=0))

    dist.mle(theta, weights)
    assert np.array_equal(dist.get_parameters(), weights.dot(theta) / np.sum(weights))

    entropy = dist.entropy()
    assert np.isclose(entropy, 6.399803705012358)

    assert np.array_equal(dist.mean(), dist.get_parameters())

    dist.con_wmle(theta, weights, 0.5)
    con_wmle_test = np.load('tests/distributions/data/test_gaussian.npy')
    assert np.allclose(dist.get_parameters(), con_wmle_test)


def test_diagonal_gaussian():
    np.random.seed(42)
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
    log_p_test = -2.599084899766805
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)
    assert np.array_equal(dist.get_parameters()[:n_dims], theta.mean(axis=0))
    assert np.array_equal(dist.get_parameters()[n_dims:], theta.std(axis=0))

    dist.mle(theta, weights)
    wmle_test = np.load('tests/distributions/data/test_diagonal_gaussian.npy')
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.487750719294691)

    assert np.array_equal(dist.mean(), dist.get_parameters()[:n_dims])


def test_cholesky_gaussian():
    np.random.seed(42)
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
    log_p_test = -6.475847829950512
    assert np.isclose(log_p, log_p_test)

    theta = np.random.randn(100, n_dims)

    weights = np.random.rand(100)

    dist.mle(theta)

    mle_test = np.load('tests/distributions/data/test_cholesky_gaussian_mle.npy')
    assert np.allclose(dist.get_parameters(), mle_test)

    dist.mle(theta, weights)
    wmle_test = np.load('tests/distributions/data/test_cholesky_gaussian_wmle.npy')
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.398170245353436)

    assert np.array_equal(dist.mean(), dist.get_parameters()[:n_dims])

    dist.con_wmle(theta, weights, 0.5, 0.1)
    con_wmle_test = np.load('tests/distributions/data/test_cholesky_gaussian_con_wmle.npy')
    assert np.allclose(dist.get_parameters(), con_wmle_test)


def test_logging():
    np.random.seed(42)
    n_dims = 3

    mu = np.zeros(n_dims)
    std = np.ones(n_dims)
    dist = GaussianDiagonalDistribution(mu, std)

    logger = FakeLogger()
    dist.set_logger(logger, 'distribution')

    theta = np.random.randn(50, n_dims)
    dist.mle(theta)

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['distribution/entropy'], dist.entropy())

    dist = GaussianDiagonalDistribution(np.zeros(2), np.ones(2))
    dist.set_parameters(np.array([1., 2., 1., 1.]))

    assert dist._logger is None
    assert np.allclose(dist.get_parameters(), np.array([1., 2., 1., 1.]))


def test_mean_copy():
    mu = np.array([1., 2.])

    for dist in [GaussianDistribution(mu.copy(), np.eye(2)),
                 GaussianDiagonalDistribution(mu.copy(), np.ones(2)),
                 GaussianCholeskyDistribution(mu.copy(), np.eye(2))]:
        mean = dist.mean()

        assert np.allclose(mean, mu)
        assert not np.shares_memory(mean, dist._mu)

        dist._mu[:] = 99.

        assert np.allclose(mean, mu)
