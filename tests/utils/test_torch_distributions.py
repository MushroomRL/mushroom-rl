import torch

import pytest

from mushroom_rl.utils.torch_utils import SquashedGaussian, CategoricalWrapper, DistHelperWrapper


def test_squashed_gaussian_bounds_and_consistency():
    low = torch.tensor([-2., -2.])
    high = torch.tensor([2., 2.])
    dist = SquashedGaussian(torch.zeros(2), torch.ones(2), low, high)

    torch.manual_seed(42)
    action, log_prob_direct = dist.rsample_and_log_prob()

    assert torch.all(action >= low) and torch.all(action <= high)
    assert log_prob_direct.shape == ()

    log_prob_external = dist.log_prob(action.detach())

    assert torch.isclose(log_prob_direct, log_prob_external, atol=1e-4)


def test_squashed_gaussian_rsample_and_log_prob():
    low = torch.tensor([-2., -2.])
    high = torch.tensor([2., 2.])
    loc = torch.zeros(2, requires_grad=True)
    scale = torch.ones(2, requires_grad=True)
    dist = SquashedGaussian(loc, scale, low, high)

    torch.manual_seed(42)
    action, log_prob = dist.rsample_and_log_prob()

    assert action.requires_grad and log_prob.requires_grad
    assert torch.allclose(action, torch.tensor([0.64903897, 0.25620341]), atol=1e-6)
    assert torch.isclose(log_prob, torch.tensor(-3.16132236), atol=1e-6)

    log_prob.backward()
    assert loc.grad is not None and scale.grad is not None


def test_squashed_gaussian_log_prob_interior():
    low = torch.tensor([-2., -2.])
    high = torch.tensor([2., 2.])
    dist = SquashedGaussian(torch.zeros(2), torch.ones(2), low, high)

    actions = torch.tensor([[0.0, 0.0], [1.0, -0.5]])
    log_prob = dist.log_prob(actions)

    assert log_prob.shape == (2,)
    assert torch.allclose(log_prob, torch.tensor([-3.22417140, -3.05543733]), atol=1e-6)


def test_squashed_gaussian_log_prob_clamps_out_of_range():
    low = torch.tensor([-2., -2.])
    high = torch.tensor([2., 2.])
    dist = SquashedGaussian(torch.zeros(2), torch.ones(2), low, high)

    above = dist.log_prob(torch.tensor([[5.0, 5.0]]))
    at_high = dist.log_prob(torch.tensor([[2.0, 2.0]]))
    below = dist.log_prob(torch.tensor([[-5.0, -5.0]]))

    assert torch.isfinite(above).all() and torch.isfinite(below).all()
    assert torch.allclose(above, at_high)
    assert torch.allclose(below, at_high)
    assert torch.allclose(at_high, torch.tensor([-29.53545380]), atol=1e-4)


def test_squashed_gaussian_boundary_is_finite():
    low = torch.tensor([-1., -1.])
    high = torch.tensor([1., 1.])
    dist = SquashedGaussian(torch.zeros(2), torch.ones(2), low, high)

    boundary_action = torch.tensor([[1.0, -1.0]])
    log_prob = dist.log_prob(boundary_action)

    assert torch.isfinite(log_prob).all()
    assert torch.allclose(log_prob, torch.tensor([-28.1491585]), atol=1e-4)


def test_squashed_gaussian_affine_term():
    loc = torch.zeros(2)
    scale = torch.ones(2)
    unit = SquashedGaussian(loc, scale, torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
    scaled = SquashedGaussian(loc, scale, torch.tensor([-2., -2.]), torch.tensor([2., 2.]))

    torch.manual_seed(0)
    _, log_prob_unit = unit.rsample_and_log_prob()
    torch.manual_seed(0)
    _, log_prob_scaled = scaled.rsample_and_log_prob()

    assert torch.isclose(log_prob_scaled, log_prob_unit - torch.log(torch.tensor(2.)) * 2, atol=1e-5)


def test_squashed_gaussian_median():
    low = torch.tensor([-2., -2.])
    high = torch.tensor([2., 2.])
    dist = SquashedGaussian(torch.tensor([0.5, -0.3]), torch.ones(2), low, high)

    median = dist.median

    assert torch.all(median >= low) and torch.all(median <= high)
    assert torch.allclose(median, torch.tensor([0.92423431, -0.58262521]), atol=1e-6)


def test_categorical_wrapper_mode():
    wrapper = CategoricalWrapper(torch.tensor([[0.1, 0.9], [0.8, 0.2]]))

    assert torch.equal(wrapper.mode, torch.tensor([1, 0]))


def test_categorical_wrapper_squeezes():
    wrapper = CategoricalWrapper(torch.tensor([[0.1, 0.9], [0.8, 0.2]]))
    log_prob = wrapper.log_prob(torch.tensor([[1], [0]]))

    assert log_prob.shape == (2,)
    assert torch.allclose(log_prob, torch.tensor([-0.37110078, -0.43748802]), atol=1e-6)


def test_dist_helper_wrapper_multivariate_normal():
    loc = torch.tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])
    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=torch.diag(torch.tensor([.5, 2.])),
                                                  validate_args=False)
    wrapper = DistHelperWrapper(dist)
    idx = torch.tensor([3, 1])
    action = torch.tensor([[1., 1.], [-1., 2.], [0., 0.], [3., -3.]])

    assert len(wrapper) == 4 and wrapper.distribution is dist
    assert type(wrapper[idx]) is torch.distributions.MultivariateNormal
    assert torch.equal(wrapper[idx].loc, loc[idx])
    assert torch.equal(wrapper[idx].log_prob(action[idx]), dist.log_prob(action)[idx])
    assert wrapper[1:3].batch_shape == (2,)


def test_dist_helper_wrapper_multivariate_normal_covariance():
    covariance = torch.stack([torch.eye(2) * (i + 1) for i in range(4)])
    dist = torch.distributions.MultivariateNormal(loc=torch.zeros(4, 2), covariance_matrix=covariance)
    idx = torch.tensor([2, 0])
    action = torch.tensor([[1., 1.], [-1., 2.], [0.5, 0.], [3., -3.]])

    selected = DistHelperWrapper(dist)[idx]

    assert torch.allclose(selected.covariance_matrix, covariance[idx], atol=1e-6)
    assert torch.allclose(selected.log_prob(action[idx]), dist.log_prob(action)[idx], atol=1e-6)


def test_dist_helper_wrapper_categorical_wrapper():
    dist = CategoricalWrapper(torch.tensor([[0.1, 0.9, 0.], [0.8, 0.2, 1.], [0., 0., 0.], [2., -1., .5]]))
    idx = torch.tensor([1, 3])
    action = torch.tensor([[1], [2], [0], [0]])

    selected = DistHelperWrapper(dist)[idx]

    assert type(selected) is CategoricalWrapper
    assert torch.allclose(selected.probs, dist.probs[idx], atol=1e-6)
    assert torch.allclose(selected.log_prob(action[idx]), dist.log_prob(action)[idx], atol=1e-6)


def test_dist_helper_wrapper_squashed_gaussian():
    low = torch.tensor([-2., -1.])
    high = torch.tensor([2., 3.])
    loc = torch.tensor([[0., 1.], [.5, -.5], [-1., 0.], [2., .1]])
    scale = torch.tensor([[1., .5], [.2, 1.], [1., 1.], [.3, .7]])
    dist = SquashedGaussian(loc, scale, low, high, eps=1e-5)
    idx = torch.tensor([0, 3])
    action = torch.tensor([[0., 1.], [1., 2.], [-1.5, 0.], [.5, -.5]])

    selected = DistHelperWrapper(dist)[idx]

    assert type(selected) is SquashedGaussian
    assert torch.equal(selected.low, low) and torch.equal(selected.high, high) and selected.eps == 1e-5
    assert torch.equal(selected.log_prob(action[idx]), dist.log_prob(action)[idx])


def test_dist_helper_wrapper_unsupported():
    with pytest.raises(NotImplementedError):
        DistHelperWrapper(torch.distributions.Normal(torch.zeros(3), torch.ones(3)))
