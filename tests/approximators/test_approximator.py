import numpy as np
import pytest

from helper.dummies import FakeLogger

from mushroom_rl.approximators.approximator import Approximator, Ensemble
from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend
from mushroom_rl.utils import TorchUtils


class BackendApproximator(Approximator):
    def __init__(self, input_shape, output_shape, backend='numpy'):
        super().__init__(input_shape, output_shape, backend=backend)


def test_ensemble_backend():
    TorchUtils.set_default_device('cpu')

    ensemble = BackendApproximator(input_shape=(2,), output_shape=(1,), n_models=3, backend='torch')

    assert isinstance(ensemble, Ensemble)
    assert ensemble.backend is TorchBackend
    assert ensemble.device == 'cpu'
    assert [(model.backend_name, model.device) for model in ensemble._models] == [('torch', 'cpu')] * 3

    ensemble = BackendApproximator(input_shape=(2,), output_shape=(1,), n_models=2)

    assert ensemble.backend is NumpyBackend
    assert ensemble.device is None
    assert [(model.backend_name, model.device) for model in ensemble._models] == [('numpy', None)] * 2


def test_interface():
    approximator = BackendApproximator((2,), (1,))
    logger = FakeLogger()
    approximator.set_logger(logger, prefix='critic', label='error')

    assert len(approximator) == 1 and approximator[0] is approximator

    with pytest.raises(NotImplementedError):
        approximator.fit(np.zeros((1, 2)), np.zeros(1))
    with pytest.raises(NotImplementedError):
        approximator.predict(np.zeros((1, 2)))

    approximator._log()
    approximator.loss_fit = None
    approximator._log()
    approximator.loss_fit = np.array([[2.]])
    approximator._log()

    assert logger.calls == [{'critic/error': np.array(2.)}]
    assert logger.calls[0]['critic/error'].shape == ()

    ensemble = BackendApproximator((2,), (1,), n_models=2, prediction='max')
    ensemble._models[0].n_actions = 3

    assert isinstance(ensemble, Ensemble) and len(ensemble) == 2
    assert all(model.input_shape == (2,) and model.output_shape == (1,) for model in ensemble)
    assert ensemble._prediction == 'max'
    assert ensemble.n_actions == 3
