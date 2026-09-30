import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator, NumpyTorchApproximator, CMAC
from mushroom_rl.approximators.parametric.networks import QNetwork, FeedForwardNetwork
from mushroom_rl.features.tiles import Tiles


class FlattenLeastSquares:
    def __init__(self, input_shape):
        self._w = None

    def fit(self, x, y):
        x = x.reshape(len(x), -1)
        self._w = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]

    def predict(self, x):
        x = x.reshape(len(x), -1)
        return np.c_[x, np.ones(len(x))] @ self._w


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


def test_q_numpy_torch_ensemble():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 2
    approximator = QApproximator(NumpyTorchApproximator, n_actions=n_actions, output_shape=(n_actions,),
                                 input_shape=(3,), n_models=2, network=QNetwork, n_features=None, n_layers=0,
                                 optimizer={'class': optim.Adam, 'params': {}}, loss=F.mse_loss,
                                 batch_size=20, quiet=True)

    s = np.random.rand(100, 3)
    a = np.random.randint(n_actions, size=(100, 1))
    q = np.random.rand(100)
    approximator.fit(s, a, q, n_epochs=5)

    x_s = np.random.rand(2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[-0.3145111, 0.42026764],
                                    [-0.33301038, 0.5050317]]))
    assert np.allclose(y, (approximator.predict(x_s, idx=0) + approximator.predict(x_s, idx=1)) / 2)

    y = approximator.predict(x_s, x_a)
    assert np.allclose(y, np.array([0.42026764, -0.33301038]))


def test_q_action_single_window():
    np.random.seed(1)

    n_actions = 2
    s = np.random.rand(100, 2, 3)
    a = np.random.randint(n_actions, size=(100, 1))
    q = np.random.rand(100)

    approximator = QApproximator(FlattenLeastSquares, n_actions=n_actions, input_shape=(2, 3))
    approximator.fit(s, a, q)

    x_s = np.random.rand(2, 2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[0.5169289841629996, 0.5145706940449885],
                                    [0.5386022102958685, 0.6170716374524995]]))
    assert np.allclose(approximator.predict(x_s, x_a), np.array([0.5145706940449885, 0.5386022102958685]))

    y_single = approximator.predict(x_s[0])
    assert y_single.shape == (n_actions,)
    assert np.allclose(y_single, y[0])

    y_single = approximator.predict(x_s[0], x_a[0])
    assert y_single.shape == ()
    assert np.allclose(y_single, y[0, x_a[0, 0]])


def test_q_simple_single_window():
    np.random.seed(1)
    torch.manual_seed(1)

    n_actions = 3
    approximator = QApproximator(NumpyTorchApproximator, n_actions=n_actions, input_shape=(2, 3),
                                 output_shape=(n_actions,), network=FeedForwardNetwork, n_features=4)

    x_s = np.random.rand(2, 2, 3)
    x_a = np.random.randint(n_actions, size=(2, 1))

    y = approximator.predict(x_s)
    assert np.allclose(y, np.array([[-0.19303003, 0.73260486, 0.7373589],
                                    [0.1122224, 0.7962094, 0.7328558]]))
    assert np.allclose(approximator.predict(x_s, x_a), np.array([-0.19303003, 0.7328558]))

    y_single = approximator.predict(x_s[0])
    assert y_single.shape == (n_actions,)
    assert np.allclose(y_single, y[0])

    y_single = approximator.predict(x_s[0], x_a[0])
    assert y_single.shape == (1,)
    assert np.allclose(y_single, y[0, x_a[0, 0]])


def test_q_approximator_requires_input_shape():
    with pytest.raises(TypeError):
        QApproximator(ExtraTreesRegressor, n_actions=2)

    with pytest.raises(TypeError):
        QApproximator(ExtraTreesRegressor, n_actions=2, n_models=2)
