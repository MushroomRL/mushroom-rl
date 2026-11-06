import numpy as np
import pytest

from helper.cases import devices

from mushroom_rl.core import Logger
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def test_predict():
    torch.manual_seed(1)

    class FlatNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] * input_shape[1], output_shape[0])

        def forward(self, x):
            return self._h(x.float().reshape(x.shape[0], -1))

    approximator = TorchApproximator(input_shape=(4, 8), output_shape=(3,), network=FlatNet)

    single = torch.rand(4, 8)
    out_single = approximator.predict(single)
    assert out_single.shape == (3,)

    batch = torch.rand(5, 4, 8)
    out_batch = approximator.predict(batch)
    assert out_batch.shape == (5, 3)

    torch.manual_seed(1)

    class KwargNet(nn.Module):
        def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] + extra_shape[0], output_shape[0])

        def forward(self, x, extra=None):
            return self._h(torch.cat([x, extra], dim=-1))

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), network=KwargNet,
                                     kwargs_shape={'extra': (3,)})

    out_single = approximator.predict(torch.rand(4), extra=torch.rand(3))
    assert out_single.shape == (2,)

    out_batch = approximator.predict(torch.rand(5, 4), extra=torch.rand(5, 3))
    assert out_batch.shape == (5, 2)

    with pytest.raises(KeyError):
        approximator.predict(torch.rand(4))


def test_diff():
    torch.manual_seed(9)

    class ConvNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._c = nn.Conv2d(input_shape[0], 2, kernel_size=4, stride=4)
            self._h = nn.Linear(2 * 5 * 5, output_shape[0])

        def forward(self, x):
            h = F.relu(self._c(x.float()))
            return self._h(h.view(h.shape[0], -1))

    approximator = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet)

    unbatched = torch.rand(4, 20, 20)
    gradient = approximator.diff(unbatched)
    assert gradient.shape == (approximator.weights_size, 6)
    assert torch.allclose(gradient, approximator.diff(unbatched.unsqueeze(0)))

    torch.manual_seed(10)

    class KwargNet(nn.Module):
        def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] + extra_shape[0], output_shape[0])

        def forward(self, x, extra=None):
            return self._h(torch.cat([x, extra], dim=-1))

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), network=KwargNet,
                                     kwargs_shape={'extra': (3,)})

    unbatched, unbatched_extra = torch.rand(4), torch.rand(3)
    gradient = approximator.diff(unbatched, extra=unbatched_extra)
    assert gradient.shape == (approximator.weights_size, 2)
    assert torch.allclose(gradient, approximator.diff(unbatched.unsqueeze(0), extra=unbatched_extra.unsqueeze(0)))

    with pytest.raises(KeyError):
        approximator.diff(unbatched)

    torch.manual_seed(14)

    class TwoHeadNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._a = nn.Linear(input_shape[0], output_shape[0][0])
            self._b = nn.Linear(input_shape[0], output_shape[1][0])

        def forward(self, x):
            return self._a(x.float()), self._b(x.float())

    single = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet)
    ensemble = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet, n_models=3)

    unbatched = torch.rand(5)
    with pytest.raises(AssertionError):
        single.diff(unbatched)

    with pytest.raises(AssertionError):
        ensemble.diff(unbatched)


def test_fit_kwargs():
    torch.manual_seed(3)

    class KwargLinear(nn.Module):
        def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
            super().__init__()
            self._wx = nn.Parameter(torch.zeros(1))
            self._we = nn.Parameter(torch.zeros(1))

        def forward(self, x, extra=None):
            return self._wx * x + self._we * extra

    n = 200
    x = torch.arange(n, dtype=torch.float32).unsqueeze(1) / n
    extra = torch.arange(n, dtype=torch.float32).flip(0).unsqueeze(1) / n
    target = 2. * x + 3. * extra

    approximator = TorchApproximator(input_shape=(1,), output_shape=(1,), network=KwargLinear,
                                     kwargs_shape={'extra': (1,)},
                                     optimizer={'class': optim.Adam, 'params': {'lr': 0.05}},
                                     loss=F.mse_loss, batch_size=16, quiet=True)

    # a minibatch that shuffled x and extra independently (instead of together) would decorrelate extra from
    # the target it is supposed to explain, since it is deliberately given the reverse order of x here, so
    # weights only converge to (2, 3) if every minibatch keeps (x_i, extra_i, target_i) aligned by row
    approximator.fit(x, target, extra=extra, n_epochs=300)

    weights = approximator.get_weights()
    assert torch.allclose(weights, torch.tensor([2., 3.]), atol=0.05), f"weights={weights}"


def test_logger(tmpdir):
    torch.manual_seed(1)

    logger = Logger('ensemble_logger', results_dir=tmpdir, use_timestamp=True, force_numpy=True)

    approximator = TorchApproximator(input_shape=(4,),
                                     output_shape=(2,), n_models=3,
                                     network=QNetwork,
                                     n_features=None,
                                     n_layers=0,
                                     optimizer={'class': optim.Adam,
                                                'params': {}}, loss=F.mse_loss,
                                     batch_size=100, quiet=True)

    approximator.set_logger(logger)

    x = torch.rand(1000, 4)
    y = torch.rand(1000, 2)

    for i in range(50):
        approximator.fit(x, y)

    loss_0 = np.load(logger.path / 'training' / 'loss_0.npy')
    loss_1 = np.load(logger.path / 'training' / 'loss_1.npy')
    loss_2 = np.load(logger.path / 'training' / 'loss_2.npy')

    assert loss_0.shape == (50,)
    assert loss_1.shape == (50,)
    assert loss_2.shape == (50,)
    assert np.isclose(loss_0[0], 0.303314387798) and np.isclose(loss_0[-1], 0.097760476172)
    assert np.isclose(loss_1[0], 0.867674708366) and np.isclose(loss_1[-1], 0.190568745136)
    assert np.isclose(loss_2[0], 1.048998713493) and np.isclose(loss_2[-1], 0.152651250362)

    torch.manual_seed(1)

    logger = Logger('approx_force_numpy', results_dir=tmpdir, use_timestamp=True, force_numpy=True)

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,),
                                     network=QNetwork, n_features=None, n_layers=0,
                                     optimizer={'class': optim.Adam, 'params': {}},
                                     loss=F.mse_loss, batch_size=100, quiet=True)
    approximator.set_logger(logger, label='critic_loss')

    x = torch.rand(100, 4)
    y = torch.rand(100, 2)
    approximator.fit(x, y)

    assert (logger.path / 'training' / 'critic_loss.npy').exists()

    torch.manual_seed(1)

    logger = Logger('approx_no_numpy', results_dir=tmpdir, use_timestamp=True)

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,),
                                     network=QNetwork, n_features=None, n_layers=0,
                                     optimizer={'class': optim.Adam, 'params': {}},
                                     loss=F.mse_loss, batch_size=100, quiet=True)
    approximator.set_logger(logger, label='critic_loss')

    x = torch.rand(100, 4)
    y = torch.rand(100, 2)
    approximator.fit(x, y)

    assert not (logger.path / 'training' / 'critic_loss.npy').exists()


@pytest.mark.parametrize('n_models', [1, 2])
def test_fit_options(n_models):
    class Net(nn.Module):
        def __init__(self, input_shape, output_shape, action_history_shape=None, **kwargs):
            super().__init__()
            self.history_shape = action_history_shape
            self.modes = list()
            self.inits = 0
            self._h = nn.Linear(input_shape[0] + action_history_shape[0], output_shape[0])
            self._dropout = nn.Dropout(.5)

        def forward(self, x, action_history=None):
            self.modes.append(self.training)
            return self._h(self._dropout(torch.cat([x.float(), action_history.float()], -1))).squeeze(-1)

        def weights_init(self):
            self.inits += 1

    torch.manual_seed(1)
    approximator = TorchApproximator(Net, (2,), (1,), optimizer={'class': optim.SGD, 'params': {'lr': .1}},
                                     loss=F.mse_loss, batch_size=4, reinitialize=True, dropout=True,
                                     n_models=n_models, action_history_shape=(3,))
    x, y, history = torch.rand(8, 2), torch.rand(8), torch.rand(8, 3)
    networks = [approximator.network] if n_models == 1 else [approximator._base_model] + \
        [model.network for model in approximator._models]
    approximator.fit(x, y, action_history=history, n_epochs=3, epsilon=1e-3, validation_split=.5)
    fit_modes = {mode for network in networks for mode in network.modes}
    for network in networks:
        network.modes.clear()
    prediction = approximator.predict(x[:2], action_history=history[:2])

    assert fit_modes == {True}
    assert {mode for network in networks for mode in network.modes} == {False}
    assert prediction.shape == (2,)
    assert all(network.history_shape == (3,) for network in networks)
    assert [network.inits for network in networks[-n_models:]] == [1] * n_models

    if n_models == 2:
        approximator.set_learning_rate(.01)

        assert isinstance(approximator.network, Net)
        assert approximator.loss is F.mse_loss
        assert len(approximator.loss_fit) == 2
        assert [model._optimizer.param_groups[0]['lr'] for model in approximator._models] == [.01, .01]


def test_ensemble_predict():
    torch.manual_seed(42)

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), n_models=3,
                                     network=QNetwork, n_features=None, n_layers=0,
                                     optimizer={'class': optim.Adam, 'params': {}},
                                     loss=F.mse_loss, batch_size=100, quiet=True)

    x = torch.rand(100, 4)
    y = torch.rand(100, 2)
    approximator.fit(x, y)

    x_test = torch.rand(5, 4)

    y_mean = approximator.predict(x_test, prediction='mean').detach().numpy()
    y_mean_exp = np.load('tests/approximators/data/test_ensemble_predict_y_mean.npy')
    assert np.allclose(y_mean, y_mean_exp)

    y_min = approximator.predict(x_test, prediction='min').detach().numpy()
    y_min_exp = np.load('tests/approximators/data/test_ensemble_predict_y_min.npy')
    assert np.allclose(y_min, y_min_exp)

    y_max = approximator.predict(x_test, prediction='max').detach().numpy()
    y_max_exp = np.load('tests/approximators/data/test_ensemble_predict_y_max.npy')
    assert np.allclose(y_max, y_max_exp)

    y_sum = approximator.predict(x_test, prediction='sum').detach().numpy()
    y_sum_exp = np.load('tests/approximators/data/test_ensemble_predict_y_sum.npy')
    assert np.allclose(y_sum, y_sum_exp)

    y_idx0 = approximator.predict(x_test, idx=0).detach().numpy()
    y_idx0_exp = np.load('tests/approximators/data/test_ensemble_predict_y_idx0.npy')
    assert np.allclose(y_idx0, y_idx0_exp)

    y_stacked = approximator.predict(x_test, prediction='all')
    assert y_stacked.shape == (3, 5, 2)
    y_stacked0_exp = np.load('tests/approximators/data/test_ensemble_predict_y_stacked0.npy')
    assert np.allclose(y_stacked[0].detach().numpy(), y_stacked0_exp)

    torch.manual_seed(3)

    class ConvNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._c = nn.Conv2d(input_shape[0], 2, kernel_size=4, stride=4)
            self._h = nn.Linear(2 * 5 * 5, output_shape[0])

        def forward(self, x):
            h = F.relu(self._c(x.float()))
            return self._h(h.view(h.shape[0], -1))

    single = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet)
    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet, n_models=3,
                                 prediction='mean')

    unbatched = torch.rand(4, 20, 20)
    assert single.predict(unbatched).shape == (6,)
    assert ensemble.predict(unbatched).shape == (6,)

    batch_one = torch.rand(1, 4, 20, 20)
    assert single.predict(batch_one).shape == (6,)
    assert ensemble.predict(batch_one).shape == (6,)

    batch_five = torch.rand(5, 4, 20, 20)
    assert single.predict(batch_five).shape == (5, 6)
    assert ensemble.predict(batch_five).shape == (5, 6)

    torch.manual_seed(4)

    single = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                               n_features=None, n_layers=0)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                                 n_features=None, n_layers=0, n_models=3, prediction='mean')

    unbatched = torch.rand(4)
    assert single.predict(unbatched).shape == (2,)
    assert ensemble.predict(unbatched).shape == (2,)

    batch_one = torch.rand(1, 4)
    assert single.predict(batch_one).shape == (2,)
    assert ensemble.predict(batch_one).shape == (2,)

    batch_five = torch.rand(5, 4)
    assert single.predict(batch_five).shape == (5, 2)
    assert ensemble.predict(batch_five).shape == (5, 2)


def test_ensemble_predict_all():
    torch.manual_seed(5)

    class FlatNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], output_shape[0])

        def forward(self, x):
            return self._h(x.float().reshape(x.shape[0], -1))

    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=FlatNet, n_models=3,
                                 prediction='all')

    unbatched = torch.rand(4, 20, 20)
    stacked = ensemble.predict(unbatched)
    assert stacked.shape == (3, 6)
    for i in range(3):
        assert torch.allclose(stacked[i], ensemble.predict(unbatched, idx=i), atol=1e-6)

    batch_one = torch.rand(1, 4, 20, 20)
    stacked_one = ensemble.predict(batch_one)
    assert stacked_one.shape == (3, 6)
    for i in range(3):
        assert torch.allclose(stacked_one[i], ensemble.predict(batch_one, idx=i), atol=1e-6)

    batch_five = torch.rand(5, 4, 20, 20)
    stacked_five = ensemble.predict(batch_five)
    assert stacked_five.shape == (3, 5, 6)
    for i in range(3):
        assert torch.allclose(stacked_five[i], ensemble.predict(batch_five, idx=i), atol=1e-6)

    assert torch.allclose(ensemble.predict(batch_five, prediction='mean'), stacked_five.mean(0))

    torch.manual_seed(6)

    class TwoHeadNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._a = nn.Linear(input_shape[0], output_shape[0][0])
            self._b = nn.Linear(input_shape[0], output_shape[1][0])

        def forward(self, x):
            return self._a(x.float()), self._b(x.float())

    single = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet)
    ensemble = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet, n_models=4,
                                 prediction='all')

    unbatched = torch.rand(5)
    head_a, head_b = single.predict(unbatched)
    assert head_a.shape == (3,) and head_b.shape == (2,)

    head_a, head_b = ensemble.predict(unbatched, prediction='mean')
    assert head_a.shape == (3,) and head_b.shape == (2,)

    head_a, head_b = ensemble.predict(unbatched)
    assert head_a.shape == (4, 3) and head_b.shape == (4, 2)
    for i in range(4):
        model_a, model_b = ensemble.predict(unbatched, idx=i)
        assert torch.allclose(head_a[i], model_a) and torch.allclose(head_b[i], model_b)

    batch_five = torch.rand(5, 5)
    head_a, head_b = single.predict(batch_five)
    assert head_a.shape == (5, 3) and head_b.shape == (5, 2)

    head_a, head_b = ensemble.predict(batch_five, prediction='sum')
    assert head_a.shape == (5, 3) and head_b.shape == (5, 2)

    head_a, head_b = ensemble.predict(batch_five)
    assert head_a.shape == (4, 5, 3) and head_b.shape == (4, 5, 2)


def test_ensemble_predict_kwargs():
    torch.manual_seed(7)

    class KwargNet(nn.Module):
        def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] + extra_shape[0], output_shape[0])

        def forward(self, x, extra=None):
            return self._h(torch.cat([x, extra], dim=-1))

    single = TorchApproximator(input_shape=(4,), output_shape=(2,), network=KwargNet,
                               kwargs_shape={'extra': (3,)})
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=KwargNet,
                                 kwargs_shape={'extra': (3,)}, n_models=3, prediction='mean')

    unbatched, unbatched_extra = torch.rand(4), torch.rand(3)
    assert single.predict(unbatched, extra=unbatched_extra).shape == (2,)
    assert ensemble.predict(unbatched, extra=unbatched_extra).shape == (2,)

    batch_five, batch_five_extra = torch.rand(5, 4), torch.rand(5, 3)
    assert single.predict(batch_five, extra=batch_five_extra).shape == (5, 2)
    assert ensemble.predict(batch_five, extra=batch_five_extra).shape == (5, 2)

    with pytest.raises(KeyError):
        ensemble.predict(unbatched)


def test_ensemble_prediction_modes():
    torch.manual_seed(15)

    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                                 n_features=None, n_layers=0, n_models=3, prediction='mean')

    batch_five = torch.rand(5, 4)
    stacked = ensemble.predict(batch_five, prediction='all')
    assert stacked.shape == (3, 5, 2)
    for i in range(3):
        assert torch.allclose(stacked[i], ensemble.predict(batch_five, idx=i), atol=1e-6)

    assert torch.allclose(ensemble.predict(batch_five), stacked.mean(0))
    assert torch.allclose(ensemble.predict(batch_five, prediction=None), stacked.mean(0))
    assert ensemble.predict(torch.rand(4), prediction='all').shape == (3, 2)

    with pytest.raises(ValueError):
        ensemble.predict(batch_five, prediction='median')

    torch.manual_seed(12)

    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                                 n_features=None, n_layers=0, n_models=3, prediction='mean')

    batch_five = torch.rand(5, 4)
    stacked = torch.stack([ensemble.predict(batch_five, idx=i) for i in range(3)])

    mean, variance = ensemble.predict(batch_five, compute_variance=True)
    assert mean.shape == (5, 2) and variance.shape == (5, 2)
    assert torch.allclose(mean, stacked.mean(0))
    assert torch.allclose(variance, stacked.var(0))

    minimum, variance_min = ensemble.predict(batch_five, prediction='min', compute_variance=True)
    assert torch.allclose(minimum, stacked.min(0).values)
    assert torch.allclose(variance_min, stacked.var(0))

    unbatched = torch.rand(4)
    mean, variance = ensemble.predict(unbatched, compute_variance=True)
    assert mean.shape == (2,) and variance.shape == (2,)

    unaggregated = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                                     n_features=None, n_layers=0, n_models=3, prediction='all')
    assert unaggregated.predict(batch_five, compute_variance=True).shape == (3, 5, 2)

    torch.manual_seed(13)

    class TwoHeadNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._a = nn.Linear(input_shape[0], output_shape[0][0])
            self._b = nn.Linear(input_shape[0], output_shape[1][0])

        def forward(self, x):
            return self._a(x.float()), self._b(x.float())

    ensemble = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet, n_models=4,
                                 prediction='all')

    batch_five = torch.rand(5, 5)
    stacked_a, stacked_b = ensemble.predict(batch_five)

    (mean_a, variance_a), (mean_b, variance_b) = ensemble.predict(batch_five, prediction='mean',
                                                                  compute_variance=True)
    assert torch.allclose(mean_a, stacked_a.mean(0)) and torch.allclose(variance_a, stacked_a.var(0))
    assert torch.allclose(mean_b, stacked_b.mean(0)) and torch.allclose(variance_b, stacked_b.var(0))

    torch.manual_seed(11)

    class FlatNet(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            self._h = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], output_shape[0])

        def forward(self, x):
            return self._h(x.float().reshape(x.shape[0], -1))

    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=FlatNet, n_models=3)

    unbatched = torch.rand(4, 20, 20)
    gradient = ensemble.diff(unbatched)
    assert gradient.shape == (3, ensemble.weights_size[1], 6)
    for i in range(3):
        assert torch.allclose(gradient[i], ensemble[i].diff(unbatched))


def test_ensemble_fit_kwargs():
    torch.manual_seed(9)

    class KwargLinear(nn.Module):
        def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
            super().__init__()
            self._wx = nn.Parameter(torch.zeros(1))
            self._we = nn.Parameter(torch.zeros(1))

        def forward(self, x, extra=None):
            return self._wx * x + self._we * extra

    n = 200
    x = torch.arange(n, dtype=torch.float32).unsqueeze(1) / n
    extra = torch.arange(n, dtype=torch.float32).flip(0).unsqueeze(1) / n
    target = 2. * x + 3. * extra

    ensemble = TorchApproximator(input_shape=(1,), output_shape=(1,), network=KwargLinear,
                                 kwargs_shape={'extra': (1,)}, n_models=2,
                                 optimizer={'class': optim.Adam, 'params': {'lr': 0.05}},
                                 loss=F.mse_loss, batch_size=16, quiet=True)

    ensemble.fit(x, target, extra=extra, n_epochs=300)

    for weights in ensemble.get_weights():
        assert torch.allclose(weights, torch.tensor([2., 3.]), atol=0.05), f"weights={weights}"


def test_ensemble_save_load(tmpdir):
    torch.manual_seed(8)

    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork,
                                 n_features=None, n_layers=0, n_models=3, prediction='mean')

    unbatched = torch.rand(4)
    expected = ensemble.predict(unbatched)

    path = str(tmpdir / 'ensemble.msh')
    ensemble.save(path, full_save=True)
    loaded = TorchApproximator.load(path)

    assert torch.allclose(loaded.predict(unbatched), expected)
    assert loaded.predict(unbatched).shape == (2,)
    assert loaded.predict(unbatched, prediction='min').shape == (2,)


def test_ensemble_full_batch():
    torch.manual_seed(7)

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), n_models=3,
                                     network=QNetwork, n_features=None, n_layers=0,
                                     optimizer={'class': optim.Adam, 'params': {}},
                                     loss=F.mse_loss, batch_size=0, quiet=True)

    x = torch.rand(50, 4)
    y = torch.rand(50, 2)

    for _ in range(10):
        approximator.fit(x, y)

    x_test = torch.rand(5, 4)
    y_out = approximator.predict(x_test, prediction='mean')
    y_out_exp = torch.tensor(np.load('tests/approximators/data/test_ensemble_full_batch.npy'))
    assert torch.allclose(y_out, y_out_exp)


@pytest.mark.parametrize('device', devices())
@pytest.mark.parametrize('n_models', [1, 3])
def test_to_device(n_models, device):
    torch.manual_seed(1)
    x, y = torch.randn(32, 3), torch.randn(32, 2)
    approximator = TorchApproximator(network=QNetwork, input_shape=(3,), output_shape=(2,), n_features=4, n_layers=1,
                                     n_models=n_models, optimizer={'class': optim.Adam, 'params': {'lr': 1e-2}},
                                     loss=F.mse_loss, device='cpu')
    approximator.fit(x, y, n_epochs=2)

    assert approximator.to(device) is approximator

    models = [approximator] if n_models == 1 else approximator._models
    assert approximator.device == device
    assert {p.device.type for p in approximator.parameters()} == {device}
    assert {value.device.type for m in models for state in m._optimizer.state.values() for value in state.values()
            if torch.is_tensor(value) and value.ndim > 0} == {device}

    approximator.fit(x.to(device), y.to(device), n_epochs=2)

    assert approximator.predict(x.to(device)).device.type == device
