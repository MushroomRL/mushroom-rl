import torch
from mushroom_rl.distributions.torch_distribution import (
    DiagonalGaussianTorchDistribution, CholeskyGaussianTorchDistribution
)


def test_diagonal_sample_log_pdf_entropy():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    sigma = torch.abs(torch.randn(n)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)

    torch.manual_seed(1)
    theta = dist.sample()
    assert torch.allclose(theta, torch.tensor([2.3244, 0.1066, -2.1477, 1.1512]), atol=1e-4)
    assert torch.isclose(dist.log_pdf(theta), torch.tensor(-3.9484), atol=1e-4)
    assert torch.isclose(dist.entropy(), torch.tensor(5.4992), atol=1e-4)


def test_diagonal_mean():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    sigma = torch.abs(torch.randn(n)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    assert torch.allclose(dist.mean(), mu)


def test_diagonal_parameters_size():
    torch.manual_seed(0)
    mu = torch.randn(4)
    sigma = torch.abs(torch.randn(4)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    assert dist.parameters_size == 8
    assert len(dist.parameters()) == 2


def test_diagonal_get_set_parameters():
    torch.manual_seed(0)
    mu = torch.randn(4)
    sigma = torch.abs(torch.randn(4)) + 0.1
    dist = DiagonalGaussianTorchDistribution(mu, sigma)
    rho = dist.get_parameters().detach().clone()
    dist.set_parameters(torch.zeros_like(rho))
    assert torch.allclose(dist._mu.data, torch.zeros(4))
    dist.set_parameters(rho)
    assert torch.allclose(dist.get_parameters().detach(), rho)


def test_cholesky_sample_log_pdf_entropy():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)

    torch.manual_seed(1)
    theta = dist.sample()
    assert torch.allclose(theta, torch.tensor([2.0947, -0.1599, -2.1404, 0.9096]), atol=1e-4)
    assert torch.isclose(dist.log_pdf(theta), torch.tensor(-4.1711), atol=1e-4)
    assert torch.isclose(dist.entropy(), torch.tensor(5.7219), atol=1e-4)


def test_cholesky_mean():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    assert torch.allclose(dist.mean(), mu)


def test_cholesky_parameters_size():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    assert dist.parameters_size == 14
    assert len(dist.parameters()) == 2


def test_cholesky_get_set_parameters():
    torch.manual_seed(0)
    n = 4
    mu = torch.randn(n)
    A = torch.randn(n, n)
    sigma = A @ A.T + torch.eye(n) * 0.1
    dist = CholeskyGaussianTorchDistribution(mu, sigma)
    rho = dist.get_parameters().detach().clone()
    dist.set_parameters(rho)
    assert torch.allclose(dist.get_parameters().detach(), rho)