import torch
import torch.nn as nn

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
