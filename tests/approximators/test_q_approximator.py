import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator, NumpyTorchApproximator, CMAC
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.features.tiles import Tiles


def test_q_cmac():
    np.random.seed(1)

    n_actions = 2
    s = np.random.rand(1000, 3)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    tilings = Tiles.generate(10, [10, 10, 10], np.zeros(3), np.ones(3))
    approximator = QApproximator(CMAC, n_actions=n_actions, n_models=5,
                                 tilings=tilings, input_shape=(3,))

    approximator.fit(s, a, q)

    x_s = np.random.rand(2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s, x_a)
    assert np.allclose(y, np.array([0.12141921, 0.20993534]))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[0.21725586, 0.12141921],
                                    [0.64425621, 0.20993534]]))


def test_q_linear():
    np.random.seed(1)

    n_actions = 2
    s = np.random.rand(1000, 3)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    approximator = QApproximator(LinearApproximator, n_actions=n_actions, input_shape=(3,))
    approximator.fit(s, a, q)

    x_s = np.random.rand(2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s, x_a)
    assert np.allclose(y, np.array([0.49040831, 0.52521538]))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[0.55168512, 0.49040831],
                                    [0.58759307, 0.52521538]]))

    approximator2 = QApproximator(LinearApproximator, n_actions=n_actions, input_shape=(3,))
    approximator2.fit(s, a, q)

    gradient = approximator2.diff(x_s[0], x_a[0])
    assert np.allclose(gradient, np.array([0., 0., 0., 0.68707335, 0.75278119, 0.33828342]))


def test_q_torch():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    s = torch.as_tensor(np.random.rand(1000, 4))
    a = torch.as_tensor(np.random.randint(n_actions, size=(1000, 1)))
    q = torch.as_tensor(np.random.rand(1000))

    approximator = QApproximator(TorchApproximator, n_actions=n_actions, output_shape=(n_actions,),
                                 input_shape=(4,), network=QNetwork, n_features=None, n_layers=0,
                                 optimizer={'class': optim.Adam, 'params': {}}, loss=F.mse_loss,
                                 batch_size=100, quiet=True)

    approximator.fit(s, a, q, n_epochs=20)

    x_s = torch.as_tensor(np.random.rand(2, 4))
    x_a = torch.as_tensor(np.random.randint(n_actions, size=(2, 1)))

    y = approximator.predict(x_s, x_a).detach().numpy()
    assert np.allclose(y, np.array([0.60176706, 0.5803694], dtype=np.float32))

    y = approximator.predict(x_s).detach().numpy()
    assert np.allclose(y, np.array([[0.60176706, 0.42583174],
                                    [0.5803694,  0.47669965]], dtype=np.float32))

    gradient = approximator.diff(x_s[0], x_a[0]).detach().numpy()
    assert np.allclose(gradient, np.array([0.45738867, 0.4335527, 0.77928376, 0.42085838,
                                           0., 0., 0., 0., 1., 0.], dtype=np.float32))

    gradient = approximator.diff(x_s[0]).detach().numpy()
    assert np.allclose(gradient, np.array([[0.45738867, 0.], [0.4335527, 0.],
                                           [0.77928376, 0.], [0.42085838, 0.],
                                           [0., 0.45738867], [0., 0.4335527],
                                           [0., 0.77928376], [0., 0.42085838],
                                           [1., 0.], [0., 1.]], dtype=np.float32))


def test_q_numpy_torch():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    s = np.random.rand(1000, 4)
    a = np.random.randint(n_actions, size=(1000, 1))
    q = np.random.rand(1000)

    approximator = QApproximator(NumpyTorchApproximator, n_actions=n_actions, output_shape=(n_actions,),
                                 input_shape=(4,), network=QNetwork, n_features=None, n_layers=0,
                                 optimizer={'class': optim.Adam, 'params': {}}, loss=F.mse_loss,
                                 batch_size=100, quiet=True)

    approximator.fit(s, a, q, n_epochs=20)

    x_s = np.random.rand(2, 4)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s, x_a)
    assert np.allclose(y, np.array([0.60176706, 0.5803694]))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[0.60176706, 0.42583174],
                                    [0.5803694,  0.47669965]]))

    gradient = approximator.diff(x_s[0], x_a[0])
    assert np.allclose(gradient, np.array([0.45738867, 0.4335527, 0.77928376, 0.42085838,
                                           0., 0., 0., 0., 1., 0.]))

    gradient = approximator.diff(x_s[0])
    assert np.allclose(gradient, np.array([[0.45738867, 0.], [0.4335527, 0.],
                                           [0.77928376, 0.], [0.42085838, 0.],
                                           [0., 0.45738867], [0., 0.4335527],
                                           [0., 0.77928376], [0., 0.42085838],
                                           [1., 0.], [0., 1.]]))
