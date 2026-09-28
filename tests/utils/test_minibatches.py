import pytest
import torch

from mushroom_rl.utils.minibatches import minibatch_generator


def test_minibatch_generator_tensors():
    x = torch.arange(10)
    y = torch.arange(10) * 10

    torch.manual_seed(1)
    batches = list(minibatch_generator(4, x, y))
    torch.manual_seed(1)
    indexes = torch.randperm(10)

    assert [len(x_i) for x_i, _ in batches] == [4, 4, 2]
    assert torch.equal(torch.cat([x_i for x_i, _ in batches]), indexes)
    assert all(torch.equal(y_i, x_i * 10) for x_i, y_i in batches)


def test_minibatch_generator_distribution():
    x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    dist = torch.distributions.MultivariateNormal(loc=x * 2, scale_tril=torch.eye(1), validate_args=False)

    torch.manual_seed(1)
    batches = list(minibatch_generator(4, x, dist))
    torch.manual_seed(1)
    batches_tensors_only = list(minibatch_generator(4, x))

    assert [dist_i.batch_shape for _, dist_i in batches] == [(4,), (4,), (2,)]
    assert all(torch.equal(x_i, x_only_i) for (x_i, _), (x_only_i,) in zip(batches, batches_tensors_only))
    assert all(torch.equal(dist_i.loc, x_i * 2) for x_i, dist_i in batches)


def test_minibatch_generator_distribution_first():
    x = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    dist = torch.distributions.MultivariateNormal(loc=x, scale_tril=torch.eye(1), validate_args=False)

    batches = list(minibatch_generator(4, dist, x))

    assert [len(x_i) for _, x_i in batches] == [4, 2]
    assert all(torch.equal(dist_i.loc, x_i) for dist_i, x_i in batches)


def test_minibatch_generator_unsupported_distribution():
    x = torch.arange(6, dtype=torch.float32)
    dist = torch.distributions.Normal(x, torch.ones(6))

    with pytest.raises(NotImplementedError):
        next(minibatch_generator(4, x, dist))
