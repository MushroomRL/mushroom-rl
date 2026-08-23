import torch
import torch.nn as nn
import torch.optim as optim

from pytest import raises

from mushroom_rl.utils.torch_utils import TorchUtils


def test_compute_output_shape_conv():
    module = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )

    output_shape = TorchUtils.compute_output_shape(module, (4, 84, 84))

    assert output_shape == (64, 7, 7)


def test_compute_output_shape_linear():
    module = nn.Linear(10, 3)

    output_shape = TorchUtils.compute_output_shape(module, (10,))

    assert output_shape == (3,)


def test_compute_flat_output_size():
    module = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )

    flat_size = TorchUtils.compute_flat_output_size(module, (4, 84, 84))

    assert flat_size == 3136


def test_compute_output_shape_does_not_consume_rng_or_mutate_module():
    module = nn.Linear(10, 3)
    weight_before = module.weight.detach().clone()
    rng_state_before = torch.get_rng_state()

    TorchUtils.compute_output_shape(module, (10,))

    assert torch.equal(module.weight, weight_before)
    assert torch.equal(torch.get_rng_state(), rng_state_before)
    assert module.weight.device == torch.device('cpu')


def test_get_optimizer():
    adam = TorchUtils.get_optimizer('adam', 1e-3, eps=1e-8)
    assert adam == {'class': optim.Adam, 'params': dict(lr=1e-3, eps=1e-8)}

    adadelta = TorchUtils.get_optimizer('adadelta', 1e-3)
    assert adadelta == {'class': optim.Adadelta, 'params': dict(lr=1e-3)}

    rmsprop = TorchUtils.get_optimizer('rmsprop', 1e-3, eps=1e-8, decay=.95)
    assert rmsprop == {'class': optim.RMSprop, 'params': dict(lr=1e-3, eps=1e-8, alpha=.95)}

    centered = TorchUtils.get_optimizer('rmspropcentered', 1e-3, eps=1e-8, decay=.95)
    assert centered == {'class': optim.RMSprop,
                        'params': dict(lr=1e-3, eps=1e-8, alpha=.95, centered=True)}


def test_get_optimizer_case_insensitive():
    assert TorchUtils.get_optimizer('Adam', 1e-3) == TorchUtils.get_optimizer('adam', 1e-3)


def test_get_optimizer_unknown():
    with raises(ValueError):
        TorchUtils.get_optimizer('sgd', 1e-3)
