import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Logger, MushroomObject
from mushroom_rl.approximators import Ensemble
from mushroom_rl.approximators.parametric import NumpyTorchApproximator, TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork


def make_approximator(**kwargs):
    return NumpyTorchApproximator(input_shape=(3,), output_shape=(2,), network=QNetwork, n_features=None,
                                  n_layers=0, optimizer={'class': optim.Adam, 'params': {}}, loss=F.mse_loss,
                                  batch_size=20, quiet=True, **kwargs)


def test_numpy_torch_approximator():
    np.random.seed(1)
    torch.manual_seed(1)

    approximator = make_approximator()

    x = np.random.rand(100, 3)
    y = np.random.rand(100, 2)
    approximator.fit(x, y, n_epochs=5)

    y_hat = approximator.predict(x[:2])
    assert isinstance(y_hat, np.ndarray)
    assert np.allclose(y_hat, np.array([[-0.099473774, 0.648228],
                                        [-0.026544452, 0.36338845]]))

    y_hat_single = approximator.predict(x[0])
    assert isinstance(y_hat_single, np.ndarray)
    assert np.allclose(y_hat_single, y_hat[0])

    w = approximator.get_weights()
    assert isinstance(w, np.ndarray)
    assert w.shape == (approximator.weights_size,)

    approximator.set_weights(np.zeros(approximator.weights_size))
    assert np.allclose(approximator.predict(x[:2]), 0.)


def test_numpy_torch_approximator_model():
    np.random.seed(1)
    torch.manual_seed(1)

    approximator = make_approximator()

    x = np.random.rand(10, 3)
    y = np.random.rand(10, 2)
    approximator.fit(x, y, n_epochs=2)

    model = approximator.model
    assert isinstance(model, TorchApproximator)

    y_torch = model.predict(torch.as_tensor(x)).detach().numpy()
    assert np.array_equal(approximator.predict(x), y_torch)
    assert np.array_equal(approximator.get_weights(), model.get_weights().detach().numpy())


def test_numpy_torch_approximator_ensemble():
    np.random.seed(1)
    torch.manual_seed(1)

    approximator = make_approximator(n_models=3)

    assert isinstance(approximator, Ensemble)
    assert len(approximator) == 3
    for i in range(3):
        assert isinstance(approximator[i], NumpyTorchApproximator)

    x = np.random.rand(100, 3)
    y = np.random.rand(100, 2)
    approximator.fit(x, y, n_epochs=5)

    y_hat = approximator.predict(x[:2])
    assert isinstance(y_hat, np.ndarray)
    assert np.allclose(y_hat, np.array([[-0.30214795, 0.38645002],
                                        [-0.18841632, -0.023845255]]))

    y_members = np.stack([approximator[i].predict(x[:2]) for i in range(3)])
    assert np.allclose(y_hat, y_members.mean(0))
    assert np.allclose(approximator.predict(x[:2], idx=1), np.array([[-0.5792092, 0.17651889],
                                                                     [-0.33411986, -0.15423518]]))
    assert np.array_equal(approximator.predict(x[:2], prediction='all'), y_members)


def test_numpy_torch_approximator_save_load(tmpdir):
    np.random.seed(1)
    torch.manual_seed(1)

    approximator = make_approximator()
    ensemble = make_approximator(n_models=2)

    x = np.random.rand(20, 3)
    y = np.random.rand(20, 2)
    approximator.fit(x, y, n_epochs=2)
    ensemble.fit(x, y, n_epochs=2)

    approximator.save(tmpdir / 'approximator')
    ensemble.save(tmpdir / 'ensemble')
    approximator_load = MushroomObject.load(tmpdir / 'approximator')
    ensemble_load = MushroomObject.load(tmpdir / 'ensemble')

    assert isinstance(approximator_load.model, TorchApproximator)
    assert np.array_equal(approximator_load.predict(x), approximator.predict(x))
    assert np.array_equal(ensemble_load.predict(x), ensemble.predict(x))


def test_numpy_torch_approximator_logger(tmpdir):
    np.random.seed(1)
    torch.manual_seed(1)

    logger = Logger('numpy_torch_logger', results_dir=tmpdir, use_timestamp=True, force_numpy=True)

    approximator = make_approximator()
    approximator.set_logger(logger, label='critic_loss')
    ensemble = make_approximator(n_models=2)
    ensemble.set_logger(logger)

    x = np.random.rand(20, 3)
    y = np.random.rand(20, 2)
    approximator.fit(x, y, n_epochs=1)
    ensemble.fit(x, y, n_epochs=1)

    assert (logger.path / 'training' / 'critic_loss.npy').exists()
    assert (logger.path / 'training' / 'loss_0.npy').exists()
    assert (logger.path / 'training' / 'loss_1.npy').exists()
