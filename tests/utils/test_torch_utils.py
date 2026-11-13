import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytest import raises

from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.torch_utils.torch_training import TorchTrainer


def test_output_shape():
    module = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )

    output_shape = TorchUtils.compute_output_shape(module, (4, 84, 84))

    assert output_shape == (64, 7, 7)

    module = nn.Linear(10, 3)

    output_shape = TorchUtils.compute_output_shape(module, (10,))

    assert output_shape == (3,)

    module = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )

    flat_size = TorchUtils.compute_flat_output_size(module, (4, 84, 84))

    assert flat_size == 3136

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

    assert TorchUtils.get_optimizer('Adam', 1e-3) == TorchUtils.get_optimizer('adam', 1e-3)

    assert TorchUtils.get_optimizer('rmsprop', 1e-3) == {'class': optim.RMSprop, 'params': dict(lr=1e-3)}

    with raises(ValueError):
        TorchUtils.get_optimizer('sgd', 1e-3)


def test_lookups():
    params = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))]
    params[1].grad = torch.full((3,), 2.)

    assert torch.equal(TorchUtils.get_gradient(params), torch.tensor([0., 0., 2., 2., 2.]))
    assert TorchUtils.get_activation('ReLU') is nn.ReLU
    assert TorchUtils.get_recurrent_network('lstm') is nn.LSTM

    for call in (lambda: TorchUtils.get_activation('unknown'), lambda: TorchUtils.get_recurrent_network('unknown'),
                 lambda: TorchUtils.init_weights(nn.Linear(2, 2), 1., weights_init='normal'),
                 lambda: TorchUtils.init_weights(nn.Linear(2, 2), 1., bias_init='ones')):
        with raises(ValueError):
            call()


def test_weights():
    params = [nn.Parameter(torch.zeros(2, 2)), nn.Parameter(torch.zeros(3))]
    weights = torch.arange(7, dtype=torch.float32)

    TorchUtils.set_weights(params, weights)
    weights.zero_()

    assert torch.equal(params[0].data, torch.tensor([[0., 1.], [2., 3.]]))
    assert torch.equal(params[1].data, torch.tensor([4., 5., 6.]))
    assert torch.equal(TorchUtils.get_weights(params), torch.arange(7, dtype=torch.float32))

    TorchUtils.set_weights(params, np.arange(7.) + .5)

    assert params[0].dtype == torch.float32
    assert torch.equal(TorchUtils.get_weights(params), torch.arange(7, dtype=torch.float32) + .5)


def make_trainer(train_losses, val_losses=None, quiet=True, loss=F.mse_loss):
    calls = dict(train=list(), val=list(), stored=list())
    train_losses, val_losses = iter(train_losses), iter(val_losses or [])

    def fit_epoch(args, use_weights, network_kwargs):
        calls['train'].append((len(args[0]), use_weights, network_kwargs))
        return torch.tensor([next(train_losses)], dtype=torch.float64)

    def val_loss(args, use_weights, network_kwargs):
        calls['val'].append(len(args[0]))
        return next(val_losses)

    trainer = TorchTrainer(loss, 2, 1, False, False, fit_epoch, val_loss, calls['stored'].append, quiet)
    return trainer, calls


@pytest.mark.parametrize('quiet', [True, False])
def test_fit(quiet):
    x = (np.zeros(4), np.zeros(4))

    trainer, calls = make_trainer([5., 4., 3.99, 3.95, 0.], quiet=quiet)
    trainer.fit(x, None, None, 0.1, 2, 1., dict(a=1))

    target = torch.tensor([[5.], [4.], [3.99], [3.95]], dtype=torch.float64)
    assert torch.equal(torch.stack(calls['stored']), target)
    assert calls['train'][0] == (4, False, dict(a=1))
    assert calls['val'] == []

    trainer, calls = make_trainer([1.] * 5, [3., 2., 1.95, 1.92, 0.])
    trainer.fit(x, None, np.ones(4), 0.1, 2, 0.5, dict())

    assert len(calls['stored']) == 4
    assert calls['train'][0] == (2, True, dict())
    assert calls['val'] == [2] * 4

    trainer, calls = make_trainer([3., 2., 1.])
    trainer.fit(x, 2, None, 0.1, 5, 1., dict())

    target = torch.tensor([[3.], [2.]], dtype=torch.float64)
    assert torch.equal(torch.stack(calls['stored']), target)

    trainer, calls = make_trainer([3., 2., 1.], quiet=quiet)
    trainer.fit(x, None, None, None, 5, 1., dict())

    target = torch.tensor([[3.]], dtype=torch.float64)
    assert torch.equal(torch.stack(calls['stored']), target)

    for split in (0., 1.5):
        with pytest.raises(ValueError):
            trainer.fit(x, None, None, None, 5, split, dict())


def test_loss():
    trainer, _ = make_trainer([])
    y_hat = torch.tensor([1., 2.], dtype=torch.float64)
    y = torch.tensor([0., 0.])
    weights = torch.tensor([1., 3.], dtype=torch.float64)

    assert trainer.loss is F.mse_loss
    assert trainer.to(None) is trainer
    assert torch.equal(trainer.compute_loss_from_output(y_hat, [y]), torch.tensor(2.5, dtype=torch.float64))
    assert torch.equal(trainer.compute_loss_from_output(y_hat, [y], weights), torch.tensor(3.25, dtype=torch.float64))

    trainer, _ = make_trainer([], loss=lambda output, target: F.mse_loss(output[0] + output[1], target))

    assert torch.equal(trainer.compute_loss_from_output((y_hat, y_hat), [y]), torch.tensor(10., dtype=torch.float64))
