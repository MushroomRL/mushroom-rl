import numpy as np
from mushroom_rl.distributions import *
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
    wmle_test = np.array([ 0.15593144, -0.09015819,  0.06310449, -0.02479729, -0.17266137,
                            0.04501165,  0.93521283,  0.93517738,  1.02358103,  0.94439444,
                            1.07237331,  1.07481608])
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.487750719294691)


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

    mle_test = np.array([-0.18129564,  0.15145523,  0.12950102, -0.14675118, -0.00414876,
                          0.02530839,  1.07648894, -0.00951914,  1.07871011,  0.05569449,
                          0.0242835 ,  0.92714112, -0.04334935, -0.00721217, -0.02253386,
                          0.93596673, -0.29276611,  0.05085615, -0.02773648, -0.1832614 ,
                          0.95865511,  0.04086696, -0.0585859 , -0.13229675, -0.03377006,
                         -0.00291481,  0.90979281])
    assert np.allclose(dist.get_parameters(), mle_test)

    dist.mle(theta, weights)
    wmle_test = np.array([-0.08694296,  0.11313395,  0.10607875, -0.04840444, -0.11541166,
                          -0.00923628,  1.1372729 , -0.04715041,  1.02481764,  0.00315469,
                          -0.10054765,  0.96970133, -0.06784433, -0.04317921, -0.00271539,
                           0.95497878, -0.31632473,  0.14081723, -0.05889656, -0.15912514,
                           0.89851531,  0.07221545, -0.09813957, -0.10400764,  0.02005758,
                          -0.07618892,  0.91872909])
    assert np.allclose(dist.get_parameters(), wmle_test)

    entropy = dist.entropy()
    assert np.isclose(entropy, 8.398170245353436)
