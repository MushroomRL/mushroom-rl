import numpy as np
import pytest

from mushroom_rl.core import Logger
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def test_predict_single_sample_multidim_input():
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


def test_predict_kwargs_shape_batch_padding_and_fail_fast():
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


def test_fit_kwargs_shape_minibatch_alignment():
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


def test_torch_ensemble_logger(tmpdir):
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


def test_torch_approximator_logger_force_numpy(tmpdir):
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


def test_torch_approximator_logger_no_numpy(tmpdir):
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


def test_torch_ensemble_predict():
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
    y_mean_exp = np.array([[3.6242208e-01, -3.8994253e-02], [-7.2021288e-04, 1.4174472e-01],
                           [7.6367331e-01, -2.5443366e-01], [7.0005685e-01, -5.4735202e-02],
                           [2.3372100e-01, 3.9389334e-03]])
    assert np.allclose(y_mean, y_mean_exp)

    y_min = approximator.predict(x_test, prediction='min').detach().numpy()
    y_min_exp = np.array([[-0.96762663, -0.7917408], [-0.77063906, 0.00266981],
                          [0.14815213, -1.1048715], [-0.17288272, -0.9199739],
                          [-0.9468446, -0.5513398]])
    assert np.allclose(y_min, y_min_exp)

    y_max = approximator.predict(x_test, prediction='max').detach().numpy()
    y_max_exp = np.array([[1.53127, 0.35895473], [0.84040344, 0.21462242],
                          [1.3279692, 0.28780013], [1.6225293, 0.58951503],
                          [1.2630175, 0.43326783]])
    assert np.allclose(y_max, y_max_exp)

    y_sum = approximator.predict(x_test, prediction='sum').detach().numpy()
    y_sum_exp = np.array([[1.0872662e+00, -1.1698276e-01], [-2.1606386e-03, 4.2523414e-01],
                          [2.2910199e+00, -7.6330101e-01], [2.1001706e+00, -1.6420561e-01],
                          [7.0116299e-01, 1.1816800e-02]])
    assert np.allclose(y_sum, y_sum_exp)

    y_idx0 = approximator.predict(x_test, idx=0).detach().numpy()
    y_idx0_exp = np.array([[1.53127, 0.35895473], [0.84040344, 0.00266981],
                           [1.3279692, 0.28780013], [1.6225293, 0.58951503],
                           [1.2630175, 0.12988877]])
    assert np.allclose(y_idx0, y_idx0_exp)

    y_stacked = approximator.predict(x_test, prediction='all')
    assert y_stacked.shape == (3, 5, 2)
    y_stacked0_exp = np.array([[1.53127, 0.35895473], [0.84040344, 0.00266981],
                               [1.3279692, 0.28780013], [1.6225293, 0.58951503],
                               [1.2630175, 0.12988877]])
    assert np.allclose(y_stacked[0].detach().numpy(), y_stacked0_exp)


def test_torch_ensemble_predict_multidim_input_matches_single_model():
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


def test_torch_ensemble_predict_1d_input_matches_single_model():
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


def test_torch_ensemble_predict_all_keeps_model_axis():
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


def test_torch_ensemble_predict_multi_output():
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


def test_torch_ensemble_predict_kwargs_shape_batch_padding():
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


def test_torch_ensemble_fit_kwargs_shape_minibatch_alignment():
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


def test_torch_ensemble_predict_after_save_load(tmpdir):
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


def test_torch_ensemble_full_batch():
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
    y_out_exp = torch.tensor([[-0.0140205, 0.49749446],
                              [0.20315261, 0.88453037],
                              [0.18861724, 0.6976128],
                              [0.045397, 0.5313599],
                              [0.41143703, 0.6998553]])
    assert torch.allclose(y_out, y_out_exp)


def test_diff_multidim_input():
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


def test_diff_kwargs_shape_batch_padding():
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


def test_torch_ensemble_diff_multidim_input():
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


def test_torch_ensemble_predict_compute_variance():
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


def test_diff_multi_output_not_supported():
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


def test_torch_ensemble_predict_all():
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


def test_torch_ensemble_predict_compute_variance_multi_output():
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
