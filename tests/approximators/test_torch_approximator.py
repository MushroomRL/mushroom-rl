import numpy as np

from mushroom_rl.core import Logger
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator, NumpyTorchApproximator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ExampleNet(nn.Module):
    def __init__(self, input_shape, output_shape,
                 **kwargs):
        super(ExampleNet, self).__init__()

        self._q = nn.Linear(input_shape[0], output_shape[0])

        nn.init.xavier_uniform_(self._q.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x, a=None):
        x = x.float()
        q = self._q(x)

        if a is None:
            return q
        else:
            action = a.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


def test_torch_approximator():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    s = torch.as_tensor(np.random.rand(1000, 4))
    a = torch.as_tensor(np.random.randint(n_actions, size=(1000, 1)))
    q = torch.as_tensor(np.random.rand(1000))

    approximator = Regressor(TorchApproximator, input_shape=(4,),
                             output_shape=(2,), n_actions=n_actions,
                             network=ExampleNet,
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

    approximator = Regressor(NumpyTorchApproximator, input_shape=(4,),
                             output_shape=(2,), n_actions=n_actions,
                             network=ExampleNet,
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
    np.random.seed(1)
    torch.manual_seed(1)

    logger = Logger('ensemble_logger', results_dir=tmpdir, use_timestamp=True)

    approximator = Regressor(TorchApproximator, input_shape=(4,),
                             output_shape=(2,), n_models=3,
                             network=ExampleNet,
                             optimizer={'class': optim.Adam,
                                        'params': {}}, loss=F.mse_loss,
                             batch_size=100, quiet=True)

    approximator.set_logger(logger)

    x = np.random.rand(1000, 4)
    y = np.random.rand(1000, 2)

    for i in range(50):
        approximator.fit(x, y)

    loss_file = np.load(logger.path / 'loss.npy')

    assert loss_file.shape == (50, 3)
    assert np.allclose(loss_file[0], np.array([0.29083753, 0.86829887, 1.0505845])) and \
           np.allclose(loss_file[-1], np.array([0.09410495, 0.18786799, 0.15016919]))
