import math
import pickle

import numpy as np
import pytest

from helper.cases import devices

from mushroom_rl.core import Logger
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork, QNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FlatNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self._h = nn.Linear(math.prod(input_shape), output_shape[0])

    def forward(self, x):
        return self._h(x.float().reshape(x.shape[0], -1))


class KwargNet(nn.Module):
    def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
        super().__init__()
        self._h = nn.Linear(input_shape[0] + extra_shape[0], output_shape[0])

    def forward(self, x, extra=None):
        return self._h(torch.cat([x, extra], dim=-1))


class ConvNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self._c = nn.Conv2d(input_shape[0], 2, kernel_size=4, stride=4)
        self._h = nn.Linear(2 * 5 * 5, output_shape[0])

    def forward(self, x):
        h = F.relu(self._c(x.float()))
        return self._h(h.view(h.shape[0], -1))


class TwoHeadNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self._a = nn.Linear(input_shape[0], output_shape[0][0])
        self._b = nn.Linear(input_shape[0], output_shape[1][0])

    def forward(self, x):
        return self._a(x.float()), self._b(x.float())


class KwargLinear(nn.Module):
    def __init__(self, input_shape, output_shape, extra_shape=None, **kwargs):
        super().__init__()
        self._wx = nn.Parameter(torch.zeros(1))
        self._we = nn.Parameter(torch.zeros(1))

    def forward(self, x, extra=None):
        return self._wx * x + self._we * extra


class ModeRecordingNet(nn.Module):
    def __init__(self, input_shape, output_shape, action_history_shape=None, **kwargs):
        super().__init__()
        self.history_shape = action_history_shape
        self.modes = list()
        self.inits = 0
        self._h = nn.Linear(input_shape[0] + action_history_shape[0], 1)
        self._dropout = nn.Dropout(.5)

    def forward(self, x, action_history=None):
        self.modes.append(self.training)
        return self._h(self._dropout(torch.cat([x.float(), action_history.float()], -1))).squeeze(-1)

    def reset_parameters(self):
        self.inits += 1


class CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer('calls', torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return super().forward(x)


class EncodedNet(nn.Module):
    def __init__(self, input_shape, output_shape, encoder=None, **kwargs):
        super().__init__()
        self._encoder = encoder
        self._h = nn.Linear(encoder.out_features, output_shape[0])

    def forward(self, x):
        return self._h(F.relu(self._encoder(x.float())))


class TiedNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0], input_shape[0])
        self._h2 = nn.Linear(input_shape[0], input_shape[0])
        self._h2.weight = self._h1.weight
        self._out = nn.Linear(input_shape[0], output_shape[0])

    def forward(self, x):
        return self._out(F.relu(self._h2(F.relu(self._h1(x.float())))))


class FrozenNet(nn.Module):
    def __init__(self, input_shape, output_shape, encoder=None, **kwargs):
        super().__init__()
        self._encoder = encoder
        self._h1 = nn.Linear(encoder.out_features, encoder.out_features)
        self._h1.weight.requires_grad_(False)
        self._h2 = nn.Linear(encoder.out_features, output_shape[0])

    def forward(self, x):
        return self._h2(F.relu(self._h1(F.relu(self._encoder(x.float())))))


class CountingNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self._h = nn.Linear(input_shape[0], output_shape[0])
        self.register_buffer('calls', torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return self._h(x.float())


def test_predict():
    torch.manual_seed(1)
    approximator = TorchApproximator(input_shape=(4, 8), output_shape=(3,), network=FlatNet)

    single = torch.rand(4, 8)
    out_single = approximator.predict(single)
    assert out_single.shape == (3,)

    batch = torch.rand(5, 4, 8)
    out_batch = approximator.predict(batch)
    assert out_batch.shape == (5, 3)

    torch.manual_seed(1)
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
    approximator = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet)

    unbatched = torch.rand(4, 20, 20)
    gradient = approximator.diff(unbatched)
    assert gradient.shape == (approximator.weights_size, 6)
    assert torch.allclose(gradient, approximator.diff(unbatched.unsqueeze(0)))

    torch.manual_seed(10)
    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), network=KwargNet,
                                     kwargs_shape={'extra': (3,)})

    unbatched, unbatched_extra = torch.rand(4), torch.rand(3)
    gradient = approximator.diff(unbatched, extra=unbatched_extra)
    assert gradient.shape == (approximator.weights_size, 2)
    assert torch.allclose(gradient, approximator.diff(unbatched.unsqueeze(0), extra=unbatched_extra.unsqueeze(0)))

    with pytest.raises(KeyError):
        approximator.diff(unbatched)

    expected = torch.cat([torch.block_diag(unbatched[None], unbatched[None]).T, torch.eye(2)])
    for shared_optimizer in [True, False]:
        torch.manual_seed(11)
        ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=LinearNetwork, use_bias=True,
                                     n_models=2, optimizer={'class': optim.SGD, 'params': {'lr': .1}},
                                     loss=F.mse_loss, shared_optimizer=shared_optimizer)
        ensemble.fit(torch.rand(8, 4), torch.rand(8, 2))

        assert torch.equal(ensemble.diff(unbatched), expected.expand(2, -1, -1))

    approximator = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork, n_features=[5], n_layers=1)

    assert approximator.diff(unbatched, torch.tensor([1])).shape == (approximator.weights_size,)
    with pytest.raises(AssertionError):
        approximator.diff(torch.rand(3, 4), torch.tensor([[1], [0], [1]]))

    torch.manual_seed(14)
    single = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet)
    ensemble = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet, n_models=3)

    unbatched = torch.rand(5)
    with pytest.raises(AssertionError):
        single.diff(unbatched)

    with pytest.raises(AssertionError):
        ensemble.diff(unbatched)


def test_fit_kwargs():
    torch.manual_seed(3)
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


@pytest.mark.parametrize('n_models, ensemble_params', [(None, {}), (2, {}), (2, {'shared_optimizer': False})])
def test_reinitialize(n_models, ensemble_params):
    def make(reinitialize):
        return TorchApproximator(QNetwork, (2,), (1,), n_features=4,
                                 optimizer={'class': optim.Adam, 'params': {'lr': .5}},
                                 loss=F.mse_loss, reinitialize=reinitialize, n_models=n_models, **ensemble_params)

    torch.manual_seed(0)
    x, y = torch.rand(8, 2), torch.rand(8, 1)
    torch.manual_seed(1)
    fresh = make(False)
    fresh.fit(x, y, n_epochs=1)
    reinitialized = make(True)
    reinitialized.fit(x, y, n_epochs=3)
    torch.manual_seed(1)
    reinitialized.fit(x, y, n_epochs=1)

    assert torch.equal(reinitialized.predict(x), fresh.predict(x))


@pytest.mark.parametrize('dropout', [True, False])
@pytest.mark.parametrize('n_models', [1, 2])
def test_fit_options(n_models, dropout):
    torch.manual_seed(1)
    approximator = TorchApproximator(ModeRecordingNet, (2,), (),
                                     optimizer={'class': optim.SGD, 'params': {'lr': .1}}, loss=F.mse_loss,
                                     batch_size=4, reinitialize=True, dropout=dropout, n_models=n_models,
                                     action_history_shape=(3,))
    x, y, history = torch.rand(8, 2), torch.rand(8), torch.rand(8, 3)
    networks = [approximator.network] if n_models == 1 else [approximator._base_model] + \
        [model.network for model in approximator._models]
    approximator.fit(x, y, action_history=history, n_epochs=3, epsilon=1e-3, validation_split=.5)
    fit_modes = {mode for network in networks for mode in network.modes}
    for network in networks:
        network.modes.clear()
    prediction = approximator.predict(x[:2], action_history=history[:2])

    assert fit_modes == {True}
    assert {mode for network in networks for mode in network.modes} == {not dropout}
    assert prediction.shape == (2,)
    assert all(network.history_shape == (3,) for network in networks)
    assert [network.inits for network in networks[-n_models:]] == [1] * n_models

    if n_models == 2:
        approximator.set_learning_rate(.01)

        assert isinstance(approximator.network, ModeRecordingNet)
        assert approximator.loss is F.mse_loss
        assert len(approximator.loss_fit) == 2
        assert approximator._optimizer.param_groups[0]['lr'] == .01


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
    single = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet)
    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=ConvNet, n_models=3,
                                 prediction='mean')

    unbatched = torch.rand(4, 20, 20)
    assert single.predict(unbatched).shape == (6,)
    assert ensemble.predict(unbatched).shape == (6,)

    batch_one = torch.rand(1, 4, 20, 20)
    assert single.predict(batch_one).shape == (1, 6)
    assert ensemble.predict(batch_one).shape == (1, 6)

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
    assert single.predict(batch_one).shape == (1, 2)
    assert ensemble.predict(batch_one).shape == (1, 2)

    batch_five = torch.rand(5, 4)
    assert single.predict(batch_five).shape == (5, 2)
    assert ensemble.predict(batch_five).shape == (5, 2)


def test_ensemble_predict_all():
    torch.manual_seed(5)
    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=FlatNet, n_models=3,
                                 prediction='all')

    unbatched = torch.rand(4, 20, 20)
    stacked = ensemble.predict(unbatched)
    assert stacked.shape == (3, 6)
    for i in range(3):
        assert torch.allclose(stacked[i], ensemble.predict(unbatched, idx=i), atol=1e-6)

    batch_one = torch.rand(1, 4, 20, 20)
    stacked_one = ensemble.predict(batch_one)
    assert stacked_one.shape == (3, 1, 6)
    for i in range(3):
        assert torch.allclose(stacked_one[i], ensemble.predict(batch_one, idx=i), atol=1e-6)

    batch_five = torch.rand(5, 4, 20, 20)
    stacked_five = ensemble.predict(batch_five)
    assert stacked_five.shape == (3, 5, 6)
    for i in range(3):
        assert torch.allclose(stacked_five[i], ensemble.predict(batch_five, idx=i), atol=1e-6)

    assert torch.allclose(ensemble.predict(batch_five, prediction='mean'), stacked_five.mean(0))

    torch.manual_seed(6)
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
    ensemble = TorchApproximator(input_shape=(5,), output_shape=[(3,), (2,)], network=TwoHeadNet, n_models=4,
                                 prediction='all')

    batch_five = torch.rand(5, 5)
    stacked_a, stacked_b = ensemble.predict(batch_five)

    (mean_a, variance_a), (mean_b, variance_b) = ensemble.predict(batch_five, prediction='mean',
                                                                  compute_variance=True)
    assert torch.allclose(mean_a, stacked_a.mean(0)) and torch.allclose(variance_a, stacked_a.var(0))
    assert torch.allclose(mean_b, stacked_b.mean(0)) and torch.allclose(variance_b, stacked_b.var(0))

    torch.manual_seed(11)
    ensemble = TorchApproximator(input_shape=(4, 20, 20), output_shape=(6,), network=FlatNet, n_models=3)

    unbatched = torch.rand(4, 20, 20)
    gradient = ensemble.diff(unbatched)
    assert gradient.shape == (3, ensemble.weights_size[1], 6)
    for i in range(3):
        assert torch.allclose(gradient[i], ensemble[i].diff(unbatched))


def test_ensemble_fit_kwargs():
    torch.manual_seed(9)
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


@pytest.mark.parametrize('shared_optimizer', [True, False])
def test_ensemble_optimizer(shared_optimizer):
    torch.manual_seed(16)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=QNetwork, n_features=8, n_layers=1,
                                 n_models=3, optimizer={'class': optim.Adam, 'params': {'lr': 1e-2}},
                                 loss=F.mse_loss, shared_optimizer=shared_optimizer)
    x, y = torch.rand(16, 4), torch.rand(16, 2)
    ensemble.fit(x, y, n_epochs=2)
    weights = ensemble.get_weights()
    fitted_parameters = ensemble.parameters() if shared_optimizer else \
        [p for model in ensemble for p in model.network.parameters()]

    assert all(p.grad is not None and not p.grad.requires_grad for p in fitted_parameters)
    assert weights.shape == ensemble.weights_size
    assert [p.shape for p in ensemble.parameters()] == [(3, *p.shape) for p in ensemble[0].network.parameters()]
    assert torch.allclose(ensemble.predict(x, prediction='all'),
                          torch.stack([ensemble.predict(x, idx=i) for i in range(3)]), atol=1e-6)

    if shared_optimizer:
        ensemble.predict(x, idx=0).sum().backward()
        ensemble.fit(x, y)

        assert all(p.grad is None for model in ensemble for p in model.network.parameters())
        with pytest.raises(AssertionError):
            ensemble.fit(x, y, idx=1)
    else:
        ensemble.set_learning_rate(.5)
        ensemble.fit(x, y, idx=1)
        fitted = ensemble.get_weights()

        assert [model._optimizer.param_groups[0]['lr'] for model in ensemble] == [.5, .5, .5]
        assert torch.equal(fitted[[0, 2]], weights[[0, 2]]) and not torch.equal(fitted[1], weights[1])
        assert torch.allclose(ensemble.predict(x, prediction='all')[1], ensemble.predict(x, idx=1), atol=1e-6)

        for idx in [None, 1]:
            ensemble.predict(x).sum().backward()
            ensemble.fit(x, y, idx=idx)

            assert all(p.grad is None for p in ensemble.parameters())


@pytest.mark.parametrize('shared_optimizer', [True, False])
def test_ensemble_shared_parameters(shared_optimizer):
    torch.manual_seed(4)
    encoder = CountingLinear(4, 5)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=EncodedNet, encoder=encoder,
                                 n_models=2, optimizer={'class': optim.SGD, 'params': {'lr': .1}}, loss=F.mse_loss,
                                 shared_optimizer=shared_optimizer)
    x, y = torch.rand(16, 4), torch.rand(16, 2)
    parameters = list(dict.fromkeys(p for model in ensemble for p in model.network.parameters()))
    loss = sum(F.mse_loss(model.network(x), y) for model in ensemble)
    expected = [(p - .1 * g).detach() for p, g in zip(parameters, torch.autograd.grad(loss, parameters))]
    ensemble.fit(x, y)
    ensemble.predict(x)

    if shared_optimizer:
        assert all(torch.allclose(p, e, atol=1e-6) for p, e in zip(parameters, expected))

    with torch.no_grad():
        encoder.weight.add_(1.)

    assert [p.shape for p in ensemble.parameters()] == [(5, 4), (5,), (2, 2, 5), (2, 2)]
    assert torch.allclose(ensemble.predict(x, prediction='all'),
                          torch.stack([ensemble.predict(x, idx=i) for i in range(2)]), atol=1e-6)

    weights = ensemble.get_weights()
    weights[0, :25], weights[1, :25] = 1., 2.
    ensemble.set_weights(weights)

    assert torch.equal(encoder.weight, torch.full((5, 4), 2.)) and torch.equal(encoder.bias, torch.full((5,), 2.))
    assert torch.equal(ensemble.get_weights()[:, 25:], weights[:, 25:])


@pytest.mark.parametrize('shared_optimizer', [True, False])
def test_ensemble_frozen_parameters(shared_optimizer):
    torch.manual_seed(20)
    encoder = nn.Linear(4, 5)
    encoder.weight.requires_grad_(False)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=FrozenNet, encoder=encoder,
                                 n_models=2, optimizer={'class': optim.Adam, 'params': {'lr': 1e-2}},
                                 loss=F.mse_loss, shared_optimizer=shared_optimizer)
    weights = ensemble.get_weights()
    ensemble.fit(torch.rand(16, 4), torch.rand(16, 2))
    changed = (ensemble.get_weights() != weights).any(0)

    assert not changed[:20].any() and not changed[25:50].any()
    assert changed[20:25].any() and changed[50:].any()


def test_ensemble_tied_weights():
    torch.manual_seed(19)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=TiedNet, n_models=2, batch_size=0,
                                 optimizer={'class': optim.SGD, 'params': {'lr': .1}}, loss=F.mse_loss)
    x, y = torch.rand(16, 4), torch.rand(16, 2)

    assert torch.allclose(ensemble.predict(x, prediction='all'),
                          torch.stack([ensemble.predict(x, idx=i) for i in range(2)]), atol=1e-6)

    expected = list()
    for model in ensemble:
        parameters = list(model.network.parameters())
        gradient = torch.autograd.grad(F.mse_loss(model.network(x), y), parameters)
        expected.append(torch.cat([(p - .1 * g).detach().flatten() for p, g in zip(parameters, gradient)]))
    ensemble.fit(x, y)

    assert torch.allclose(ensemble.get_weights(), torch.stack(expected), atol=1e-6)
    assert all(model.network._h1.weight is model.network._h2.weight for model in ensemble)


@pytest.mark.parametrize('shared_optimizer', [True, False])
def test_ensemble_storage(shared_optimizer, tmpdir):
    torch.manual_seed(18)
    ensemble = TorchApproximator(input_shape=(4,), output_shape=(2,), network=CountingNet, n_models=2,
                                 optimizer={'class': optim.Adam, 'params': {'lr': 1e-2}}, loss=F.mse_loss,
                                 shared_optimizer=shared_optimizer)
    x, y = torch.rand(16, 4), torch.rand(16, 2)
    ensemble.predict(x)
    ensemble.predict(x, idx=1)

    assert [model.network.calls.item() for model in ensemble] == [1., 2.]

    ensemble.fit(x, y)
    copied = ensemble.copy()
    pickled = pickle.loads(pickle.dumps(ensemble))
    path = str(tmpdir / 'ensemble.msh')
    ensemble.save(path)
    loaded = TorchApproximator.load(path)
    for approximator in [ensemble, copied, pickled, loaded]:
        approximator.fit(x, y)

        assert torch.allclose(approximator.predict(x, prediction='all'),
                              torch.stack([approximator.predict(x, idx=i) for i in range(2)]), atol=1e-6)

    assert torch.equal(copied.get_weights(), ensemble.get_weights())
    assert torch.equal(pickled.get_weights(), ensemble.get_weights())
    assert torch.equal(loaded.get_weights(), ensemble.get_weights())
    with pytest.raises(AssertionError):
        ensemble[0].to('cpu')
    with pytest.raises(NotImplementedError):
        ensemble[0].save(path)


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
    network = approximator.network
    network(x).sum().backward()

    assert approximator.to(device) is approximator

    assert approximator.device == device
    if n_models > 1:
        assert {model.device for model in approximator} == {device}
    assert {p.device.type for p in approximator.parameters()} == {device}
    assert approximator.network is network
    assert {t.device.type for p in network.parameters() for t in (p, p.grad)} == {device}
    assert {value.device.type for state in approximator._optimizer.state.values() for value in state.values()
            if torch.is_tensor(value) and value.ndim > 0} == {device}

    approximator.fit(x.to(device), y.to(device), n_epochs=2)

    assert approximator.predict(x.to(device)).device.type == device
