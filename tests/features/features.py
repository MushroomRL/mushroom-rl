import numpy as np

from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.features.basis import GaussianRBF, FourierBasis
from mushroom.features.tensors import gaussian_tensor


def tiles():
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


def basis():
    rbf = GaussianRBF.generate([3, 3], [[0., 1.], [-.5, .5]])
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


def tensor():
    rbf = gaussian_tensor.generate([3, 3], [[0., 1.], [-.5, .5]])
    features = Features(tensor_list=rbf, name='rbf', input_dim=2)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    for i, x_i in enumerate(x):
        assert np.allclose(features(x_i), y[i])

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.allclose(features(x_1, x_2), y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.allclose(features(x_i[0], x_i[1]), y[i])


def basis_and_tensors():
    basis_rbf = GaussianRBF.generate([3, 3], [[0., 1.], [-.5, .5]])
    tensor_rbf = gaussian_tensor.generate([3, 3], [[0., 1.], [-.5, .5]])
    features_1 = Features(tensor_list=tensor_rbf, name='rbf', input_dim=2)
    features_2 = Features(basis_list=basis_rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y_1 = features_1(x)
    y_2 = features_2(x)

    assert np.allclose(y_1, y_2)


def fourier():
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




if __name__ == '__main__':
    print('Executing features test...')

    tiles()
    basis()
    tensor()
    basis_and_tensors()
    fourier()
