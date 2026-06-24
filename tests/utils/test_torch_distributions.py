import torch

from mushroom_rl.utils.torch_distributions import SquashedGaussian, CategoricalWrapper


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


def test_categorical_wrapper_squeezes():
    wrapper = CategoricalWrapper(torch.tensor([[0.1, 0.9], [0.8, 0.2]]))
    log_prob = wrapper.log_prob(torch.tensor([[1], [0]]))

    assert log_prob.shape == (2,)
    assert torch.allclose(log_prob, torch.tensor([-0.37110078, -0.43748802]), atol=1e-6)
