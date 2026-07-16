import numpy as np

import pytest
import torch

from mushroom_rl.features import Features
from mushroom_rl.features._impl import BasisFeatures, TilesFeatures, TorchFeatures, FunctionalFeatures
from mushroom_rl.features.tiles import Tiles, VoronoiTiles
from mushroom_rl.features.basis import GaussianRBF, FourierBasis, PolynomialBasis
from mushroom_rl.features.tensors import GaussianRBFTensor, VonMisesTensor, RandomFourierTensor, ConstantTensor
from mushroom_rl.utils.torch_utils import TorchUtils


def test_tiles():
    np.random.seed(1)
    tilings = Tiles.generate(3, [3, 3],
                             np.array([0., -.5]),
                             np.array([1., .5]))
    features = Features(tilings)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    for i, x_i in enumerate(x):
        assert np.all(features(x_i) == y[i])

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.all(features(x_1, x_2) == y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.all(features(x_i[0], x_i[1]) == y[i])
        assert features.size == y[i].size


def test_tiles_voronoi():
    np.random.seed(1)
    tilings_list = [
        VoronoiTiles.generate(3, 10,
                              low=np.array([0., -.5]),
                              high=np.array([1., .5])),
        VoronoiTiles.generate(3, 10,
                              mu=np.array([.5, -.5]),
                              sigma=np.array([.2, .6]))
    ]

    for tilings in tilings_list:
        features = Features(tilings)

        x = np.random.rand(10, 2) + [0., -.5]

        y = features(x)

        for i, x_i in enumerate(x):
            assert np.all(features(x_i) == y[i])

        x_1 = x[:, 0].reshape(-1, 1)
        x_2 = x[:, 1].reshape(-1, 1)

        assert np.all(features(x_1, x_2) == y)

        for i, x_i in enumerate(zip(x_1, x_2)):
            assert np.all(features(x_i[0], x_i[1]) == y[i])
            assert features.size == y[i].size


def test_tiles_dimensions():
    np.random.seed(1)

    x = np.random.rand(4, 3)
    prototypes = np.random.rand(5, 2)

    tiles = Tiles([[0., 1.], [-1., 2.], [0., 1.]], [3, 3], dimensions=[0, 2])
    voronoi = VoronoiTiles(prototypes, dimensions=[0, 2])

    assert np.array_equal(tiles(x), np.array([1, 0, 3, 7]))
    assert np.array_equal(tiles(x), Tiles([[0., 1.], [0., 1.]], [3, 3])(x[:, [0, 2]]))
    assert np.array_equal(voronoi(x), VoronoiTiles(prototypes)(x[:, [0, 2]]))


def test_tiles_dimensions_mismatch():
    np.random.seed(1)

    prototypes = np.random.rand(5, 2)

    with pytest.raises(AssertionError):
        Tiles([[0., 1.], [0., 1.]], [3, 3], dimensions=[0, 1, 2])

    with pytest.raises(AssertionError):
        VoronoiTiles(prototypes, dimensions=[0, 1, 2])

    with pytest.raises(AssertionError):
        Tiles([[0., 1.], [0., 1.], [0., 1.]], [3, 3, 3], dimensions=[0, 2])


def test_tiles_generate_dimensions():
    np.random.seed(1)

    x = np.random.rand(4, 3)
    low, high = np.array([0., -1., 0.]), np.array([1., 2., 1.])

    tilings = Tiles.generate(3, [3, 3], low, high, dimensions=[0, 2])
    reference = Tiles.generate(3, [3, 3], low[[0, 2]], high[[0, 2]])

    assert len(tilings) == 3

    for tiling, reference_tiling in zip(tilings, reference):
        assert tiling.size == 9
        assert np.array_equal(tiling(x), reference_tiling(x[:, [0, 2]]))

    features = Features(tilings)

    assert features.size == 27
    assert features(x).shape == (4, 27)

    with pytest.raises(AssertionError):
        Tiles.generate(3, [3, 3, 3], low, high, dimensions=[0, 2])


def test_basis_generate_dimensions():
    np.random.seed(1)

    x = np.random.rand(4, 3)
    low, high = np.array([0., -9., 0.]), np.array([1., 9., 1.])

    generators = [
        lambda a, b, d: GaussianRBF.generate([3, 3], a, b, dimensions=d),
        lambda a, b, d: FourierBasis.generate(a, b, 2, dimensions=d),
        lambda a, b, d: GaussianRBFTensor.generate([3, 3], a, b, dimensions=d)
    ]

    for generator in generators:
        features = Features(generator(low, high, [0, 2]))
        reference = Features(generator(low[[0, 2]], high[[0, 2]], None))

        assert features.size == 9
        assert np.allclose(np.asarray(features(x)), np.asarray(reference(x[:, [0, 2]])))

    with pytest.raises(AssertionError):
        GaussianRBF.generate([3, 3, 3], low, high, dimensions=[0, 2])


def test_voronoi_generate_dimensions():
    np.random.seed(1)

    x = np.random.rand(4, 3)

    tilings = VoronoiTiles.generate(2, 5, low=np.array([0., -9., 0.]), high=np.array([1., 9., 1.]),
                                    dimensions=[0, 2])

    assert len(tilings) == 2

    for tiling in tilings:
        assert tiling.size == 5

    features = Features(tilings)

    assert features.size == 10
    assert features(x).shape == (4, 10)

    np.random.seed(2)
    gaussian = VoronoiTiles.generate(2, 5, mu=np.zeros(2), sigma=np.ones(2), dimensions=[0, 2])
    np.random.seed(2)
    reference = VoronoiTiles.generate(2, 5, mu=np.zeros(2), sigma=np.ones(2))

    for tiling, reference_tiling in zip(gaussian, reference):
        assert np.array_equal(tiling(x), reference_tiling(x[:, [0, 2]]))

    with pytest.raises(AssertionError):
        VoronoiTiles.generate(2, 5, mu=np.zeros(3), sigma=np.ones(3), dimensions=[0, 2])


def test_tiles_outside():
    tilings = Tiles.generate(2, [3, 3],
                             np.array([0., 0.]),
                             np.array([1., 1.]))
    features = Features(tilings)

    x = np.array([[.5, .5], [10., 10.], [-10., .5]])

    y = features(x)

    assert np.all(tilings[0](x) == np.array([4, -1, -1]))
    assert np.array_equal(y.sum(-1), np.array([2., 0., 0.]))


def test_polynomials():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    pol = PolynomialBasis.generate(3, 2, low, high)
    features = Features(pol)

    x = np.random.rand(3, 2) + [0., -.5]

    y = features(x)

    y_test = np.array([[1., -0.16595599, 0.44064899, 0.02754139, -0.07312834, 0.19417153, -0.00457066, 0.01213609,
                        -0.03222393,  0.08556149],
                       [1., -0.99977125, -0.39533485, 0.99954255, 0.39524442, 0.15628965, -0.99931391, -0.39515401,
                        -0.1562539,  -0.06178675],
                       [1., -0.70648822, -0.81532281, 0.4991256, 0.57601596, 0.66475129, -0.35262636, -0.40694849,
                        -0.46963895, -0.54198689]])

    assert np.allclose(y, y_test)


def test_basis():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBF.generate([3, 3], high, low)
    features = Features(rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    for i, x_i in enumerate(x):
        assert np.all(features(x_i) == y[i])

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.all(features(x_1, x_2) == y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.all(features(x_i[0], x_i[1]) == y[i])
        assert features.size == y[i].size


def test_basis_dimensions():
    np.random.seed(1)
    rbf = GaussianRBF(np.array([.5, .5]), np.array([1., 1.]), dimensions=[0, 2])

    x = np.random.rand(3, 3)

    assert np.allclose(rbf(x), np.array([0.77354535, 0.81443535, 0.89665007]))
    assert np.allclose(rbf(x), np.exp(-np.sum((x[:, [0, 2]] - .5)**2, axis=-1)))


def test_fourier_dimensions():
    np.random.seed(1)
    fourier = FourierBasis(np.array([0., 0.]), np.array([1., 1.]), np.array([1., 2.]), dimensions=[0, 2])

    x = np.random.rand(3, 3)

    assert np.allclose(fourier(x), np.array([0.25704617, 0.04079871, -0.99798611]))


def test_polynomial_dimensions():
    np.random.seed(1)
    pol = PolynomialBasis(dimensions=[0, 2], degrees=[1, 2])

    x = np.random.rand(3, 3)

    assert np.allclose(pol(x), np.array([5.45531457e-09, 2.57781331e-03, 2.93219073e-02]))
    assert np.allclose(pol(x), x[:, 0] * x[:, 2]**2)


def test_polynomial_dimensions_normalized():
    np.random.seed(1)
    pol = PolynomialBasis(dimensions=[0, 2], degrees=[1, 2],
                          low=np.array([0., 0., 0.]), high=np.array([2., 10., 100.]))

    x = np.random.rand(3, 3)

    x_n = (x - np.array([1., 5., 50.])) / np.array([1., 5., 50.])

    assert np.allclose(pol(x), np.array([-0.58297533, -0.69509294, -0.80087641]))
    assert np.allclose(pol(x), x_n[:, 0] * x_n[:, 2]**2)


def test_polynomial_constant():
    np.random.seed(1)
    pol = PolynomialBasis()

    x = np.random.rand(3, 3)

    assert np.allclose(pol(x), np.ones(3))


def test_generate_torch_bounds():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    low_torch = TorchUtils.to_float_tensor(low)
    high_torch = TorchUtils.to_float_tensor(high)

    x = np.random.rand(3, 2) + [0., -.5]

    generators = [
        lambda a, b: GaussianRBF.generate([2, 2], a, b),
        lambda a, b: FourierBasis.generate(a, b, 2),
        lambda a, b: PolynomialBasis.generate(2, 2, a, b),
        lambda a, b: GaussianRBFTensor.generate([2, 2], a, b),
        lambda a, b: Tiles.generate(2, [3, 3], a, b)
    ]

    for generate in generators:
        assert np.allclose(Features(generate(low_torch, high_torch))(x), Features(generate(low, high))(x))

    np.random.seed(1)
    voronoi_torch = VoronoiTiles.generate(2, 5, low=low_torch, high=high_torch)
    np.random.seed(1)
    voronoi_numpy = VoronoiTiles.generate(2, 5, low=low, high=high)

    assert np.allclose(Features(voronoi_torch)(x), Features(voronoi_numpy)(x))


def test_basis_repr():
    low = np.array([0., 0.])
    high = np.array([1., 1.])

    assert repr(GaussianRBF(np.array([.5, .5]), np.array([1., 1.]))) == 'GaussianRBF(mean=[0.5 0.5], scale=[1. 1.])'
    assert repr(GaussianRBF(np.array([.5, .5]), np.array([1., 1.]), dimensions=[0, 2])) == \
           'GaussianRBF(mean=[0.5 0.5], scale=[1. 1.], dimensions=[0, 2])'
    assert repr(FourierBasis(low, high, np.array([1., 2.]))) == 'FourierBasis(c=[1. 2.])'
    assert repr(PolynomialBasis()) == 'PolynomialBasis(1)'
    assert repr(PolynomialBasis(dimensions=[0, 2], degrees=[1, 2])) == 'PolynomialBasis(x[0]*x[2]^2)'

    assert repr(GaussianRBF.generate([2, 2], low, high)[0]) == \
           'GaussianRBF(mean=[0. 0.], scale=[0.08680556 0.08680556])'


def test_tiles_repr():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    assert repr(Tiles([[0., 1.], [-.5, .5]], [3, 3])) == 'Tiles(n_tiles=[3, 3], range=[[0.0, 1.0], [-0.5, 0.5]])'
    assert repr(Tiles([[0., 1.], [2., 3.], [-.5, .5]], [3, 3], dimensions=[0, 2])) == \
           'Tiles(n_tiles=[3, 3], range=[[0.0, 1.0], [-0.5, 0.5]], dimensions=[0, 2])'
    assert repr(Tiles.generate(2, [3, 3], low, high)[0]) == \
           'Tiles(n_tiles=[3, 3], range=[[-0.2, 1.0], [-0.7, 0.5]])'
    assert repr(VoronoiTiles.generate(2, 10, low=low, high=high)[0]) == 'VoronoiTiles(n_prototypes=10)'


def test_tensor_repr():
    low = np.array([0., 0.])
    high = np.array([1., 1.])

    assert repr(GaussianRBFTensor.generate([3, 3], low, high)[0]) == \
           'GaussianRBFTensor(n_centers=9, normalized=False)'
    assert repr(VonMisesTensor.generate([3, 3], low, high, normalized=True)[0]) == \
           'VonMisesTensor(n_centers=9, normalized=True)'
    assert repr(RandomFourierTensor.generate(0.1, 6, 2)[0]) == 'RandomFourierTensor(n_output=5, nu=0.1)'
    assert repr(ConstantTensor()) == 'ConstantTensor()'


def test_tensor():
    np.random.seed(1)
    torch.manual_seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBFTensor.generate([3, 3], low, high)
    rbf += VonMisesTensor.generate([3, 3], low, high, normalized=True)
    rbf += RandomFourierTensor.generate(0.1, 6, 2)

    features = Features(rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y = features(x)

    assert y.shape == (10, 24)

    for i, x_i in enumerate(x):
        assert np.allclose(features(x_i), y[i])
        assert features.size == y[i].size

    assert np.all(y[:, -1] == 1)

    x_1 = x[:, 0].reshape(-1, 1)
    x_2 = x[:, 1].reshape(-1, 1)

    assert np.allclose(features(x_1, x_2), y)

    for i, x_i in enumerate(zip(x_1, x_2)):
        assert np.allclose(features(x_i[0], x_i[1]), y[i])
        assert features.size == y[i].size


def test_basis_and_tensors():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    basis_rbf = GaussianRBF.generate([3, 3], low, high)
    tensor_rbf = GaussianRBFTensor.generate([3, 3], low, high)
    features_1 = Features(tensor_rbf)
    features_2 = Features(basis_rbf)

    x = np.random.rand(10, 2) + [0., -.5]

    y_1 = features_1(x)
    y_2 = features_2(x)

    assert np.allclose(y_1, y_2)


def test_fourier():
    low = np.array([-1.0, 0.5])
    high = np.array([1.0, 2.5])
    basis_list = FourierBasis.generate(low, high, 5)

    features = Features(basis_list)

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
    assert features.size == res.size


def test_random_fourier():
    torch.manual_seed(1)
    tensor_list = RandomFourierTensor.generate(nu=2.5, n_output=10, input_size=2)

    x = np.array([0.1, 1.4])

    features = Features(tensor_list)

    res = np.array([-0.68119127, -0.33233675,  0.9187717, -0.46273828,  0.85892403,
                    -0.947241,  0.1585586, -0.98503137,  0.5480738,  1.])

    assert np.allclose(features(x), res)
    assert features.size == res.size


def test_dispatch():
    low = np.array([0., -.5])
    high = np.array([1., .5])

    assert isinstance(Features(GaussianRBF.generate([3, 3], low, high)), BasisFeatures)
    assert isinstance(Features(Tiles.generate(3, [3, 3], low, high)), TilesFeatures)
    assert isinstance(Features(GaussianRBFTensor.generate([3, 3], low, high)), TorchFeatures)
    assert isinstance(Features(n_outputs=2), FunctionalFeatures)


def test_single_feature():
    np.random.seed(1)

    x = np.random.rand(4, 2) + [0., -.5]

    features_list = [
        Features(Tiles([[0., 1.], [-.5, .5]], [3, 3])),
        Features(GaussianRBF(np.array([.5, .5]), np.array([1., 1.]))),
        Features(ConstantTensor())
    ]

    for features in features_list:
        assert features(x).shape == (4, features.size)
        assert features(x[0]).shape == (features.size,)


def test_batch_shape_all_features():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    x = np.random.rand(4, 2) + [0., -.5]

    features_list = [
        Features(GaussianRBF.generate([3, 3], low, high)),
        Features(Tiles.generate(2, [3, 3], low, high)),
        Features(GaussianRBFTensor.generate([3, 3], low, high)),
        Features(n_outputs=2),
        Features(n_outputs=1, function=lambda z: np.sum(z, axis=-1, keepdims=True))
    ]

    for features in features_list:
        assert features(x).shape == (4, features.size)
        assert features(x[:1]).shape == (features.size,)
        assert features(x[0]).shape == (features.size,)
        assert np.allclose(np.asarray(features(x[:1])), np.asarray(features(x[0])))


def test_dispatch_errors():
    low = np.array([0., -.5])
    high = np.array([1., .5])

    rbf = GaussianRBF.generate([3, 3], low, high)[0]
    rbf_tensor = GaussianRBFTensor.generate([3, 3], low, high)[0]

    for bad_args, bad_kwargs in [(([rbf, rbf_tensor],), {}), (([],), {}), ((), {}),
                                 (([rbf],), dict(n_outputs=5)), (([rbf],), dict(function=lambda z: z))]:
        try:
            Features(*bad_args, **bad_kwargs)
            assert False
        except ValueError:
            pass


def test_functional():
    features = Features(n_outputs=3)

    x = np.random.rand(5, 3)

    assert np.allclose(features(x), x)
    assert features.size == 3

    doubling = Features(n_outputs=3, function=lambda z: 2 * z)

    assert np.allclose(doubling(x), 2 * x)


def test_backend_all_features():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    x = np.random.rand(4, 2) + [0., -.5]

    feature_args = [
        (GaussianRBF.generate([3, 3], low, high), dict()),
        (Tiles.generate(2, [3, 3], low, high), dict()),
        (GaussianRBFTensor.generate([3, 3], low, high), dict()),
        (None, dict(n_outputs=2))
    ]

    for feature_list, kwargs in feature_args:
        y_numpy = Features(feature_list, backend='numpy', **kwargs)(x)
        y_torch = Features(feature_list, backend='torch', **kwargs)(x)

        assert isinstance(y_numpy, np.ndarray)
        assert isinstance(y_torch, torch.Tensor)
        assert np.allclose(y_numpy, y_torch.numpy())


def test_torch_input_all_features():
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    x = np.random.rand(4, 2) + [0., -.5]

    feature_args = [
        (GaussianRBF.generate([3, 3], low, high), dict()),
        (Tiles.generate(2, [3, 3], low, high), dict()),
        (GaussianRBFTensor.generate([3, 3], low, high), dict()),
        (None, dict(n_outputs=2))
    ]

    for feature_list, kwargs in feature_args:
        features = Features(feature_list, backend='torch', **kwargs)

        x_torch = TorchUtils.to_float_tensor(x)

        assert torch.allclose(features(x), features(x_torch))
        assert torch.allclose(features(x), features(x_torch.clone().requires_grad_()))


def test_tensor_torch_input():
    torch.manual_seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBFTensor.generate([3, 3], low, high)

    features = Features(rbf, backend='torch')

    x = np.random.rand(4, 2) + [0., -.5]

    x_torch = TorchUtils.to_float_tensor(x)

    assert torch.allclose(features(x), features(x_torch))
    assert torch.allclose(features(x), features(x_torch.clone().requires_grad_()))

    x_1 = x_torch[:, 0].unsqueeze(-1).requires_grad_()
    x_2 = x_torch[:, 1].unsqueeze(-1).requires_grad_()

    assert torch.allclose(features(x), features(x_1, x_2))


def test_backend_not_supported():
    low = np.array([0., -.5])
    high = np.array([1., .5])

    features = Features(GaussianRBF.generate([3, 3], low, high), backend='list')

    try:
        features(np.random.rand(4, 2))
        assert False
    except NotImplementedError:
        pass


def test_torch_backend():
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBFTensor.generate([3, 3], low, high)

    x = np.random.rand(5, 2)

    numpy_features = Features(rbf)
    torch_features = Features(rbf, backend='torch')

    y_numpy = numpy_features(x)
    y_torch = torch_features(x)

    assert isinstance(y_numpy, np.ndarray)
    assert isinstance(y_torch, torch.Tensor)
    assert np.allclose(y_numpy, y_torch.numpy())


def test_to_torch_module():
    torch.manual_seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])
    rbf = GaussianRBFTensor.generate([3, 3], low, high)

    features = Features(rbf)
    module = features.to_torch_module()

    assert isinstance(module, torch.nn.Module)

    x = torch.rand(4, 2, requires_grad=True)
    y = module(x)

    assert y.shape == (4, 9)

    y.sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0.)


def test_to_torch_module_sequence():
    torch.manual_seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    tensor_list = GaussianRBFTensor.generate([3, 3], low, high) + VonMisesTensor.generate([3, 3], low, high) \
        + RandomFourierTensor.generate(2.5, 4, 2)

    features = Features(tensor_list)
    module = features.to_torch_module()

    x = torch.rand(4, 7, 2)
    y = module(x)

    assert y.shape == (4, 7, features.size)
    assert torch.allclose(y, module(x.reshape(-1, 2)).reshape(4, 7, features.size))


def test_to_torch_module_not_implemented():
    low = np.array([0., -.5])
    high = np.array([1., .5])

    for features in [Features(GaussianRBF.generate([3, 3], low, high)),
                     Features(Tiles.generate(3, [3, 3], low, high)),
                     Features(n_outputs=2)]:
        try:
            features.to_torch_module()
            assert False
        except NotImplementedError:
            pass


def test_get_action_features():
    phi_state = np.array([1., 2.])
    action = np.array([1])

    phi_state_action = Features.get_action_features(phi_state, action, 3)

    assert np.allclose(phi_state_action, np.array([0., 0., 1., 2., 0., 0.]))


def test_get_action_features_batch():
    phi_state = np.array([[1., 2.], [3., 4.]])
    action = np.array([[0], [2]])

    phi_state_action = Features.get_action_features(phi_state, action, 3)

    assert np.allclose(phi_state_action, np.array([[1., 2., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 3., 4.]]))


def test_get_action_features_single_sample_batch():
    action = np.array([[1]])

    assert np.allclose(Features.get_action_features(np.array([[1., 2.]]), action, 3),
                       np.array([0., 0., 1., 2., 0., 0.]))
    assert np.allclose(Features.get_action_features(np.array([1., 2.]), action, 3),
                       np.array([0., 0., 1., 2., 0., 0.]))


def test_get_action_features_torch():
    phi_state_action = Features.get_action_features(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[0], [2]]), 3)

    assert isinstance(phi_state_action, torch.Tensor)
    assert torch.allclose(phi_state_action, torch.tensor([[1., 2., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 3., 4.]]))

    assert torch.allclose(Features.get_action_features(torch.tensor([1., 2.]), torch.tensor([1]), 3),
                          torch.tensor([0., 0., 1., 2., 0., 0.]))
    assert torch.allclose(Features.get_action_features(torch.tensor([[1., 2.]]), torch.tensor([[1]]), 3),
                          torch.tensor([0., 0., 1., 2., 0., 0.]))


def test_serialization(tmpdir):
    np.random.seed(1)
    low = np.array([0., -.5])
    high = np.array([1., .5])

    x = np.random.rand(6, 2) + [0., -.5]

    features_list = [
        Features(GaussianRBF.generate([3, 3], low, high)),
        Features(FourierBasis.generate(low, high, 2)),
        Features(PolynomialBasis.generate(2, 2, low, high)),
        Features(Tiles.generate(3, [3, 3], low, high)),
        Features(GaussianRBFTensor.generate([3, 3], low, high)),
        Features(n_outputs=2)
    ]

    for i, features in enumerate(features_list):
        path = tmpdir / f'features_{i}.msh'
        features.save(path)
        loaded = Features.load(path)

        assert type(loaded) is type(features)
        assert loaded.size == features.size
        assert np.allclose(np.asarray(features(x)), np.asarray(loaded(x)))
