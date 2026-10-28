from mushroom_rl.approximators.approximator import Approximator, Ensemble
from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend
from mushroom_rl.utils import TorchUtils


class BackendApproximator(Approximator):
    def __init__(self, input_shape, output_shape, backend='numpy'):
        super().__init__(input_shape, output_shape, backend=backend)


def test_ensemble_takes_backend_and_device_from_its_models():
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
