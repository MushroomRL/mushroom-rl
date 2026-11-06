import numpy as np
import torch
import pytest

from helper.dummies import FakeLogger

from mushroom_rl.core.array_backend import TorchBackend
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter, CovarianceParameter, \
    FixedCovarianceParameter, DiagonalCovarianceParameter, CorrelatedCovarianceParameter


def test_values():
    sigma = np.array([[1., .3], [.3, .5]])
    c = FixedCovarianceParameter(sigma)

    assert np.array_equal(c.get_value(), sigma)
    assert np.array_equal(c.get_cholesky(), np.linalg.cholesky(sigma))
    assert np.array_equal(c.get_inverse(), np.linalg.inv(sigma))
    assert np.array_equal(c.get_std(), np.sqrt(np.array([1., .5])))

    cholesky = c.get_cholesky()
    inverse = c.get_inverse()
    value = c()

    assert np.array_equal(value, sigma)
    assert c.get_cholesky() is cholesky
    assert c.get_inverse() is inverse

    c = DiagonalCovarianceParameter(torch.tensor([.3, .2]), backend='torch')

    assert torch.allclose(c.get_value(), torch.tensor([[.09, 0.], [0., .04]]))
    assert torch.allclose(c.get_cholesky(), torch.tensor([[.3, 0.], [0., .2]]))
    assert torch.allclose(c.get_inverse(), torch.tensor(np.load('tests/rl_utils/data/test_values_inverse_torch.npy')))
    assert torch.allclose(c.get_std(), torch.tensor([.3, .2]))

    std = LinearParameter(np.array([1., .5]), np.array([.2, .1]), n=4)
    c = CorrelatedCovarianceParameter(std, np.array([[1., .5], [.5, 1.]]))

    assert np.allclose(c.get_std(), np.array([1., .5]))

    c.update()
    c.update()

    assert np.allclose(c.get_std(), np.array([.6, .3]))
    assert np.allclose(c.get_value(), np.array([[.36, .09], [.09, .09]]))
    assert np.allclose(c.get_cholesky(), np.load('tests/rl_utils/data/test_values_cholesky.npy'))
    assert np.allclose(c.get_inverse(), np.load('tests/rl_utils/data/test_values_inverse.npy'))

    value = c()

    assert np.allclose(value, np.array([[.16, .04], [.04, .04]]))
    assert np.allclose(c.get_cholesky() @ c.get_cholesky().T, value)
    assert np.allclose(c.get_inverse() @ value, np.eye(2))


def test_make():
    sigma = np.array([[1., .3], [.3, .5]])

    c = CovarianceParameter.make(sigma)
    assert isinstance(c, FixedCovarianceParameter)
    assert np.array_equal(c.get_value(), sigma)

    c = CovarianceParameter.make(np.array([.3, .2]))
    assert isinstance(c, DiagonalCovarianceParameter)
    assert np.allclose(c.get_value(), np.array([[.09, 0.], [0., .04]]))

    c = CovarianceParameter.make(Parameter(np.array([.3, .2])), backend='torch')
    assert isinstance(c, DiagonalCovarianceParameter)
    assert isinstance(c.get_value(), torch.Tensor)
    assert torch.allclose(c.get_std(), torch.tensor([.3, .2]))

    c_numpy = FixedCovarianceParameter(sigma)
    c_torch = CovarianceParameter.make(c_numpy, backend='torch')
    assert c_torch is c_numpy
    assert isinstance(c_torch.get_value(), torch.Tensor)
    assert torch.allclose(c_torch.get_cholesky(), torch.tensor(np.linalg.cholesky(sigma), dtype=torch.float32))

    correlation = np.array([[1., .5], [.5, 1.]])

    c = CovarianceParameter.make(np.array([.3, .2]), correlation=correlation)
    assert isinstance(c, CorrelatedCovarianceParameter)
    assert np.allclose(c.get_value(), np.array([[.09, .03], [.03, .04]]))

    std = LinearParameter(np.array([1., .5]), np.array([.2, .1]), n=4)
    c = CovarianceParameter.make(std, backend='torch', correlation=correlation)
    assert isinstance(c, CorrelatedCovarianceParameter)
    assert isinstance(c.get_value(), torch.Tensor)

    with pytest.raises(AssertionError):
        CovarianceParameter.make(np.eye(2), correlation=correlation)

    with pytest.raises(AssertionError):
        CovarianceParameter.make(FixedCovarianceParameter(np.eye(2)), correlation=correlation)

    with pytest.raises(TypeError):
        CovarianceParameter.make(np.array([.3, .2]), corelation=correlation)


def test_backend():
    c = CorrelatedCovarianceParameter(np.array([.3, .2]), np.array([[1., .5], [.5, 1.]]))
    assert isinstance(c.get_cholesky(), np.ndarray)

    c.set_backend('torch')
    assert c.get_cholesky().device == torch.device(TorchBackend.check_device(None))
    assert torch.allclose(c.get_value(), torch.tensor([[.09, .03], [.03, .04]]))

    c.set_backend('numpy')
    assert isinstance(c.get_value(), np.ndarray)
    assert np.allclose(c.get_value(), np.array([[.09, .03], [.03, .04]]))

    sigma = np.eye(2)

    with pytest.raises(AssertionError):
        FixedCovarianceParameter(sigma, backend='list')

    with pytest.raises(AssertionError):
        DiagonalCovarianceParameter(np.ones(2), backend='list')

    with pytest.raises(AssertionError):
        FixedCovarianceParameter(sigma).set_backend('list')

    c = CovarianceParameter(None)
    for method in [c.get_value, c.get_cholesky, c.get_inverse, c.get_std]:
        with pytest.raises(NotImplementedError):
            method()


def test_updates():
    logger = FakeLogger()
    c = DiagonalCovarianceParameter(LinearParameter(np.array([1., .5]), np.array([.2, .1]), n=4), backend='torch')
    c.set_logger(logger, 'sigma')

    c.update()

    assert len(logger.calls) == 1
    assert list(logger.calls[0].keys()) == ['sigma/value']
    assert isinstance(logger.calls[0]['sigma/value'], np.ndarray)
    assert np.allclose(logger.calls[0]['sigma/value'], np.array([.8, .4]))

    std = LinearParameter(np.array([1., .5]), np.array([.2, .1]), n=4)
    c = DiagonalCovarianceParameter(std)

    c.update()
    c.set_updates(False)
    c.update()

    assert np.allclose(c.get_std(), np.array([.8, .4]))

    c.set_updates()
    std.set_updates(False)
    c.update()

    assert np.allclose(c.get_std(), np.array([.8, .4]))


def test_serialization(tmpdir):
    path = tmpdir / 'covariance.msh'

    c = FixedCovarianceParameter(np.array([[1., .3], [.3, .5]]), backend='torch')
    c.get_cholesky()
    c.save(path)
    c_load = CovarianceParameter.load(path)

    assert torch.equal(c_load.get_value(), c.get_value())
    assert torch.equal(c_load.get_cholesky(), c.get_cholesky())

    std = LinearParameter(np.array([1., .5]), np.array([.2, .1]), n=4)
    c = CorrelatedCovarianceParameter(std, np.array([[1., .5], [.5, 1.]]))
    c.update()
    c.save(path)
    c_load = CovarianceParameter.load(path)

    assert np.array_equal(c_load.get_value(), c.get_value())
    c.update()
    c_load.update()
    assert np.array_equal(c_load.get_cholesky(), c.get_cholesky())
