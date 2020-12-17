import numpy as np

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles, VoronoiTiles
from mushroom_rl.features.basis import GaussianRBF, FourierBasis
from mushroom_rl.features.tensors import GaussianRBFTensor, RandomFourierBasis


def test_tiles():
    tilings = Tiles.generate(3, [3, 3],
                             np.array([0., -.5]),
                             np.array([1., .5]))
    features = Features(tilings=tilings)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    for i, x_i in enumerate(x):
        assert np.all(features(x_i) == y[i])

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.all(features(x_1, x_2) == y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.all(features(x_i[0], x_i[1]) == y[i])


def test_tiles_voronoi():
    tilings_list = [
        VoronoiTiles.generate(3, 10,
                              low=np.array([0., -.5]),
                              high=np.array([1., .5])),
        VoronoiTiles.generate(3, 10,
                              mu=np.array([.5, -.5]),
                              sigma=np.array([.2, .6]))
    ]

    for tilings in tilings_list:
        features = Features(tilings=tilings)

        x = np.random.rand(10, 2) + [0., -.5]

        y = features(x)

        for i, x_i in enumerate(x):
            assert np.all(features(x_i) == y[i])

        x_1 = x[:, 0].reshape(-1, 1)
        x_2 = x[:, 1].reshape(-1, 1)

        assert np.all(features(x_1, x_2) == y)

        for i, x_i in enumerate(zip(x_1, x_2)):
            assert np.all(features(x_i[0], x_i[1]) == y[i])


def test_basis():
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBF.generate([3, 3], high, low)
    features = Features(basis_list=rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    for i, x_i in enumerate(x):
        assert np.all(features(x_i) == y[i])

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.all(features(x_1, x_2) == y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.all(features(x_i[0], x_i[1]) == y[i])


def test_tensor():
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBFTensor.generate([3, 3], low, high)
    rbf += RandomFourierBasis.generate(0.1, 6, 2)
    features = Features(tensor_list=rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    assert y.shape == (10, 15)

    for i, x_i in enumerate(x):
        assert np.allclose(features(x_i), y[i])

    assert np.all(y[:, -1] == 1)

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.allclose(features(x_1, x_2), y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.allclose(features(x_i[0], x_i[1]), y[i])


def test_basis_and_tensors():
    low = np.array([0., -.5])
    high = np.array([1., .5])
    basis_rbf = GaussianRBF.generate([3, 3], low, high)
    tensor_rbf = GaussianRBFTensor.generate([3, 3], low, high)
    features_1 = Features(tensor_list=tensor_rbf)
    features_2 = Features(basis_list=basis_rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y_1 = features_1(x)
    y_2 = features_2(x)

    assert np.allclose(y_1, y_2)


def test_fourier():
    low = np.array([-1.0, 0.5])
    high = np.array([1.0, 2.5])
    basis_list = FourierBasis.generate(low, high, 5)

    features = Features(basis_list=basis_list)

    x = np.array([0.1, 1.4])

    res = np.array([1., -0.15643447, -0.95105652,  0.4539905, 0.80901699,
                    -0.70710678, 0.15643447, -1., 0.15643447, 0.95105652,
                    -0.4539905, -0.80901699, -0.95105652, -0.15643447, 1.,
                    -0.15643447, -0.95105652, 0.4539905, -0.4539905,
                    0.95105652, 0.15643447, -1., 0.15643447, 0.95105652,
                    0.80901699, 0.4539905, -0.95105652, -0.15643447,  1.,
                    -0.15643447, 0.70710678, -0.80901699, -0.4539905,
                    0.95105652, 0.15643447, -1.])

    assert np.allclose(features(x), res)


def test_random_fourier():
    np.random.seed(1)
    tensor_list = RandomFourierBasis.generate(nu=2.5, n_output=10, input_size=2)

    x = np.array([0.1, 1.4])

    features = Features(tensor_list=tensor_list)

    res = np.array([0.33279073,  0.84292346,  0.03078904, -0.98234737,  0.8367746,
                    0.476112,  0.4179958,  0.99205977,  0.5216869,  1.])

    assert np.allclose(features(x), res)
