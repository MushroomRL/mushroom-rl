import numpy as np
import pytest
import torch

from mushroom_rl.distributions.torch_distribution import (
    AbstractGaussianTorchDistribution, DiagonalGaussianTorchDistribution, CholeskyGaussianTorchDistribution
)


def test_diagonal():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    sigma = torch.abs(torch.randn(n)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)

    torch.manual_seed(1)
    theta = dist.sample()
    assert torch.allclose(theta, torch.tensor(np.load('tests/distributions/data/test_diagonal.npy')), atol=1e-4)
    assert torch.isclose(dist.log_pdf(theta), torch.tensor(-3.9484), atol=1e-4)
    assert torch.equal(dist(theta), torch.exp(dist.log_pdf(theta)))
    assert torch.isclose(dist.entropy(), torch.tensor(5.4992), atol=1e-4)

    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    sigma = torch.abs(torch.randn(n)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    assert torch.allclose(dist.mean(), mu)

    torch.manual_seed(0)
    mu = torch.randn(4)
    sigma = torch.abs(torch.randn(4)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    assert dist.parameters_size == 8
    assert len(dist.parameters()) == 2

    torch.manual_seed(0)
    mu = torch.randn(4)
    sigma = torch.abs(torch.randn(4)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    rho = dist.get_parameters().detach().clone()
    dist.set_parameters(torch.zeros_like(rho))
    assert torch.allclose(dist._mu.data, torch.zeros(4))
    dist.set_parameters(rho)
    assert torch.allclose(dist.get_parameters().detach(), rho)


def test_cholesky():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)

    torch.manual_seed(1)
    theta = dist.sample()
    assert torch.allclose(theta, torch.tensor(np.load('tests/distributions/data/test_cholesky.npy')), atol=1e-4)
    assert torch.isclose(dist.log_pdf(theta), torch.tensor(-4.1711), atol=1e-4)
    assert torch.isclose(dist.entropy(), torch.tensor(5.7219), atol=1e-4)

    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    assert torch.allclose(dist.mean(), mu)

    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    assert dist.parameters_size == 14
    assert len(dist.parameters()) == 2

    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    rho = dist.get_parameters().detach().clone()
    dist.set_parameters(rho)
    assert torch.allclose(dist.get_parameters().detach(), rho)


def test_mean_copy():
    mu = torch.tensor([1., 2.])

    for dist in [DiagonalGaussianTorchDistribution(mu.clone(), torch.ones(2)),
                 CholeskyGaussianTorchDistribution(mu.clone(), torch.eye(2))]:
        mean = dist.mean()

        assert torch.allclose(mean, mu)
        assert mean.data_ptr() != dist._mu.data_ptr()

        with torch.no_grad():
            dist._mu[:] = 99.

        assert torch.allclose(mean, mu)


def test_abstract():
    calls = [lambda: AbstractGaussianTorchDistribution.mle(None, None),
             lambda: AbstractGaussianTorchDistribution.con_wmle(None, None, None, None, None),
             lambda: AbstractGaussianTorchDistribution.diff_log(None, None),
             lambda: AbstractGaussianTorchDistribution._get_mean_and_chol(None, None),
             lambda: AbstractGaussianTorchDistribution.parameters(None)]

    for call in calls:
        with pytest.raises(NotImplementedError):
            call()
