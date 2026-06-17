import numpy as np

from mushroom_rl.core import Logger
from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import TorchApproximator, NumpyTorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork

import torch
import torch.optim as optim
import torch.nn.functional as F


def test_torch_approximator():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    s = torch.as_tensor(np.random.rand(1000, 4))
    a = torch.as_tensor(np.random.randint(n_actions, size=(1000, 1)))
    q = torch.as_tensor(np.random.rand(1000))

    approximator = QApproximator(TorchApproximator, n_actions=n_actions, output_shape=(n_actions,),
                                 input_shape=(4,),
                                 network=QNetwork,
                                 n_features=None,
                                 n_layers=0,
                                 optimizer={'class': optim.Adam,
                                            'params': {}}, loss=F.mse_loss,
                                 batch_size=100, quiet=True)

    approximator.fit(s, a, q, n_epochs=20)

    x_s = torch.as_tensor(np.random.rand(2, 4))
    x_a = torch.as_tensor(np.random.randint(n_actions, size=(2, 1)))
    y = approximator.predict(x_s, x_a).detach().numpy()
    y_test = torch.as_tensor(np.array([0.37191153, 0.5920861]))

    assert np.allclose(y, y_test)

    y = approximator.predict(x_s).detach().numpy()
    y_test = np.array([[0.47908658, 0.37191153],
                       [0.5920861, 0.27575058]])

    assert np.allclose(y, y_test)

    gradient = approximator.diff(x_s[0], x_a[0]).detach().numpy()
    gradient_test = np.array([0., 0., 0., 0., 0.02627479, 0.76513696,
                              0.6672573, 0.35979462, 0., 1.])
    assert np.allclose(gradient, gradient_test)

    gradient = approximator.diff(x_s[0])
    gradient_test = np.array([[0.02627479, 0.], [0.76513696, 0.],
                              [0.6672573, 0.], [0.35979462, 0.],
                              [0., 0.02627479], [0., 0.76513696],
                              [0., 0.6672573], [0., 0.35979462], [1, 0.],
                              [0., 1.]])
    assert np.allclose(gradient, gradient_test)

    old_weights = approximator.get_weights().detach().numpy()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights().detach().numpy()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights().detach().numpy()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))


def test_numpy_torch_approximator():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    s = np.random.rand(1000, 4)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    approximator = QApproximator(NumpyTorchApproximator, n_actions=n_actions, output_shape=(n_actions,),
                                 input_shape=(4,),
                                 network=QNetwork,
                                 n_features=None,
                                 n_layers=0,
                                 optimizer={'class': optim.Adam,
                                            'params': {}}, loss=F.mse_loss,
                                 batch_size=100, quiet=True)

    approximator.fit(s, a, q, n_epochs=20)

    x_s = np.random.rand(2, 4)
    x_a = np.random.randint(n_actions, size=(2, 1))
    y = approximator.predict(x_s, x_a)
    y_test = np.array([0.37191153, 0.5920861])

    assert np.allclose(y, y_test)

    y = approximator.predict(x_s)
    y_test = np.array([[0.47908658, 0.37191153],
                       [0.5920861, 0.27575058]])

    assert np.allclose(y, y_test)

    gradient = approximator.diff(x_s[0], x_a[0])
    gradient_test = np.array([0., 0., 0., 0., 0.02627479, 0.76513696,
                              0.6672573, 0.35979462, 0., 1.])
    assert np.allclose(gradient, gradient_test)

    gradient = approximator.diff(x_s[0])
    gradient_test = np.array([[0.02627479, 0.], [0.76513696, 0.],
                              [0.6672573, 0.], [0.35979462, 0.],
                              [0., 0.02627479], [0., 0.76513696],
                              [0., 0.6672573], [0., 0.35979462], [1, 0.],
                              [0., 1.]])
    assert np.allclose(gradient, gradient_test)

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))


def test_torch_ensemble_logger(tmpdir):
    torch.manual_seed(1)

    logger = Logger('ensemble_logger', results_dir=tmpdir, use_timestamp=True)

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

    loss_0 = np.load(logger.path / 'loss_0.npy')
    loss_1 = np.load(logger.path / 'loss_1.npy')
    loss_2 = np.load(logger.path / 'loss_2.npy')

    assert loss_0.shape == (50,)
    assert loss_1.shape == (50,)
    assert loss_2.shape == (50,)
    assert np.isclose(loss_0[0], 0.303314387798) and np.isclose(loss_0[-1], 0.097760476172)
    assert np.isclose(loss_1[0], 0.867674708366) and np.isclose(loss_1[-1], 0.190568745136)
    assert np.isclose(loss_2[0], 1.048998713493) and np.isclose(loss_2[-1], 0.152651250362)


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
