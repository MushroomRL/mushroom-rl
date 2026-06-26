import numpy as np

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
    y_mean_exp = np.array([[ 3.6242208e-01, -3.8994253e-02], [-7.2021288e-04,  1.4174472e-01],
                           [ 7.6367331e-01, -2.5443366e-01], [ 7.0005685e-01, -5.4735202e-02],
                           [ 2.3372100e-01,  3.9389334e-03]])
    assert np.allclose(y_mean, y_mean_exp)

    y_min = approximator.predict(x_test, prediction='min').detach().numpy()
    y_min_exp = np.array([[-0.96762663, -0.7917408 ], [-0.77063906,  0.00266981],
                          [ 0.14815213, -1.1048715 ], [-0.17288272, -0.9199739 ],
                          [-0.9468446 , -0.5513398 ]])
    assert np.allclose(y_min, y_min_exp)

    y_max = approximator.predict(x_test, prediction='max').detach().numpy()
    y_max_exp = np.array([[1.53127   , 0.35895473], [0.84040344, 0.21462242],
                          [1.3279692 , 0.28780013], [1.6225293 , 0.58951503],
                          [1.2630175 , 0.43326783]])
    assert np.allclose(y_max, y_max_exp)

    y_sum = approximator.predict(x_test, prediction='sum').detach().numpy()
    y_sum_exp = np.array([[ 1.0872662e+00, -1.1698276e-01], [-2.1606386e-03,  4.2523414e-01],
                          [ 2.2910199e+00, -7.6330101e-01], [ 2.1001706e+00, -1.6420561e-01],
                          [ 7.0116299e-01,  1.1816800e-02]])
    assert np.allclose(y_sum, y_sum_exp)

    y_idx0 = approximator.predict(x_test, idx=0).detach().numpy()
    y_idx0_exp = np.array([[1.53127   ,  0.35895473], [0.84040344,  0.00266981],
                           [1.3279692 ,  0.28780013], [1.6225293 ,  0.58951503],
                           [1.2630175 ,  0.12988877]])
    assert np.allclose(y_idx0, y_idx0_exp)

    y_stacked = approximator.predict(x_test)
    assert y_stacked.shape == (3, 5, 2)
    y_stacked0_exp = np.array([[1.53127   ,  0.35895473], [0.84040344,  0.00266981],
                               [1.3279692 ,  0.28780013], [1.6225293 ,  0.58951503],
                               [1.2630175 ,  0.12988877]])
    assert np.allclose(y_stacked[0].detach().numpy(), y_stacked0_exp)


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
    y_out_exp = torch.tensor([[-0.0140205 ,  0.49749446],
                               [ 0.20315261,  0.88453037],
                               [ 0.18861724,  0.6976128 ],
                               [ 0.045397  ,  0.5313599 ],
                               [ 0.41143703,  0.6998553 ]])
    assert torch.allclose(y_out, y_out_exp)
