import numpy as np

from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def test_linear_approximator():
    np.random.seed(88)

    noise = 1e-3

    a = np.random.rand(1000, 3)

    k = np.random.rand(3, 2)
    b = a.dot(k) + np.random.randn(1000, 2) * noise

    approximator = Regressor(LinearApproximator,
                             input_shape=(3,),
                             output_shape=(2,))

    approximator.fit(a, b)

    khat = approximator.get_weights()

    deltaK = (khat - k.T.flatten())

    assert np.linalg.norm(deltaK) < noise

    point = np.random.randn(3,)
    derivative = approximator.diff(point)

    lp = len(point)
    for i in range(derivative.shape[1]):
        assert (derivative[i * lp:(i + 1) * lp, i] == point).all()


class ExampleNet(nn.Module):
    def __init__(self, input_shape, output_shape, n_neurons, n_hidden,
                 **kwargs):
        super(ExampleNet, self).__init__()

        self._h = nn.ModuleList()

        h = nn.Linear(input_shape[0], n_neurons)
        self._h.append(h)

        for i in range(n_hidden):
            h = nn.Linear(n_neurons, n_neurons)
            self._h.append(h)

        h = nn.Linear(n_neurons, output_shape[0])
        self._h.append(h)

        for h in self._h[:-1]:
            nn.init.xavier_uniform_(h.weight,
                                    gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self._h[-1].weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        o = x.float()

        for h in self._h[:-1]:
            o = F.relu(h(o))

        return self._h[-1](o)


def test_pytorch_approximator():
    np.random.seed(88)
    torch.manual_seed(88)

    noise = 1e-3**2

    a = np.random.rand(1000, 4)

    k = np.random.rand(4, 2)
    b = np.sin(a).dot(k) + np.random.randn(1000, 2)*noise

    approximator = Regressor(PyTorchApproximator,
                             input_shape=(4,),
                             output_shape=(2,),
                             network=ExampleNet,
                             optimizer={'class': optim.Adam,
                                        'params': {}},
                             loss=F.mse_loss,
                             n_neurons=100,
                             n_hidden=1,
                             n_epochs=200,
                             batch_size=100,
                             quiet=True)

    approximator.fit(a, b)

    bhat = approximator.predict(a)
    error = np.linalg.norm(b - bhat, 'fro') / 1000
    error_inf = np.max(np.abs(b-bhat))

    print(b[:10])

    print(bhat[:10])

    print(error_inf)

    assert error < 2e-4

    gradient = approximator.diff(a[0])
    assert gradient.shape[1] == 2

    old_weights = approximator.get_weights()
    approximator.set_weights(old_weights)
    new_weights = approximator.get_weights()

    assert np.array_equal(new_weights, old_weights)

    random_weights = np.random.randn(*old_weights.shape).astype(np.float32)
    approximator.set_weights(random_weights)
    random_weight_new = approximator.get_weights()

    assert np.array_equal(random_weights, random_weight_new)
    assert not np.any(np.equal(random_weights, old_weights))

    bhat_random = approximator.predict(a)

    assert not np.array_equal(bhat, bhat_random)




