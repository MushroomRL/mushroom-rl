import numpy as np
import pytest
import torch

from helper.dummies import FakeLogger

from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter, DecayParameter
from mushroom_rl.rl_utils.parameters import VarianceDrivenIncreasingParameter
from mushroom_rl.policy import EpsGreedy, Mellowmax
from mushroom_rl.approximators.table import Table


def test_logging():
    logger = FakeLogger()
    p = Parameter(0.5)
    p.set_logger(logger, 'lr')

    value = p()

    assert np.isclose(value, 0.5)
    assert logger.calls == [{'lr/value': 0.5}]

    logger = FakeLogger()
    p = Parameter(0.3)
    p.set_logger(logger)

    p()

    assert logger.calls == [{'value': 0.3}]

    logger = FakeLogger()
    p = Parameter(0.5, shape=(3, 3))
    p.set_logger(logger, 'lr')

    p(1, 2)

    assert logger.calls == []

    logger = FakeLogger()
    p = Parameter(0.5, shape=(3, 3), log_full=True)
    p.set_logger(logger, 'lr')

    p(1, 2)

    assert logger.calls == [{'lr/value': 0.5}]

    logger = FakeLogger()
    p = Parameter(0.5, shape=(1, 1))
    p.set_logger(logger, 'lr')

    p()

    assert logger.calls == [{'lr/value': 0.5}]

    logger = FakeLogger()
    p = LinearParameter(1.0, threshold_value=0.0, n=10)
    p.set_logger(logger, 'eps')

    p()

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['eps/value'], 0.9)

    logger = FakeLogger()
    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3)
    p.set_logger(logger, 'std')

    p()

    assert len(logger.calls) == 1
    assert isinstance(logger.calls[0]['std/value'], np.ndarray)
    assert np.allclose(logger.calls[0]['std/value'], np.array([.7, .4]))

    logger = FakeLogger()
    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3, backend='torch')
    p.set_logger(logger, 'std')

    p()

    assert isinstance(logger.calls[0]['std/value'], np.ndarray)
    assert np.allclose(logger.calls[0]['std/value'], np.array([.7, .4]))

    logger = FakeLogger()
    p = DecayParameter(1.0, exp=1.0)
    p.set_logger(logger, 'eps')

    p()
    p()

    assert len(logger.calls) == 2
    assert np.isclose(logger.calls[0]['eps/value'], 1.0)
    assert np.isclose(logger.calls[1]['eps/value'], 0.5)

    p = Parameter(0.5)
    value = p()

    assert p._logger is None
    assert np.isclose(value, 0.5)

    logger = FakeLogger()
    p = VarianceDrivenIncreasingParameter(value=0.5, tol=1.)
    p.set_logger(logger, 'lr')

    p.update(0, target=1.0)

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['lr/value'], 0.5)

    logger = FakeLogger()
    pi = EpsGreedy(Parameter(0.1))
    pi.set_logger(logger)

    pi.update()

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['policy/epsilon'], 0.1)

    logger = FakeLogger()
    pi = Mellowmax(Parameter(3.))
    pi.set_logger(logger)

    assert logger.calls == []


def test_vector():
    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3)
    values = [p() for _ in range(5)]
    values_test = [[.7, .4], [.4, .7], [.1, 1.], [.1, 1.], [.1, 1.]]

    assert np.allclose(values, values_test)

    with pytest.raises(AssertionError):
        LinearParameter(np.array([1., .1]), .5, n=4)

    with pytest.raises(AssertionError):
        LinearParameter(1., np.array([.1, 1.]), n=4)

    device = TorchBackend.check_device(None)

    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3, backend='torch')
    values = [p() for _ in range(4)]

    assert all(isinstance(v, torch.Tensor) and v.device == torch.device(device) for v in values)
    assert torch.allclose(torch.stack(values).cpu(), torch.tensor([[.7, .4], [.4, .7], [.1, 1.], [.1, 1.]]))

    p = DecayParameter(np.array([1., 2.]), min_value=np.array([.1, .5]), backend='torch')
    values = torch.stack([p() for _ in range(3)]).cpu()

    assert torch.allclose(values, torch.tensor(np.load('tests/rl_utils/data/test_vector.npy')))

    p = Parameter(np.array([.3, .2]), backend='torch')

    assert torch.allclose(p().cpu(), torch.tensor([.3, .2]))

    assert isinstance(LinearParameter(1., .1, 3, backend='torch')(), np.float64)
    assert isinstance(Parameter(64, backend='torch')(), int)

    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3)
    p()

    p.set_backend('torch')
    value = p()

    assert isinstance(value, torch.Tensor)
    assert torch.allclose(value.cpu(), torch.tensor([.4, .7]))

    p.set_backend('numpy')
    value = p()

    assert isinstance(value, np.ndarray)
    assert np.allclose(value, np.array([.1, 1.]))

    p = Parameter.make(0.5, backend='torch')
    assert p._backend is TorchBackend

    q = Parameter(0.5)
    assert Parameter.make(q, backend='torch')._backend is TorchBackend

    q.set_backend('numpy')
    assert q._backend is NumpyBackend


def test_serialization(tmpdir):
    path = tmpdir / 'parameter.msh'

    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3, backend='torch')
    p()
    p.save(path)
    p_load = LinearParameter.load(path)

    assert torch.equal(p_load(), p())
    assert torch.equal(p_load(), p())

    p = LinearParameter(np.array([1., .1]), np.array([.1, 1.]), n=3)
    p()
    p.save(path)
    p_load = LinearParameter.load(path)

    assert isinstance(p_load.initial_value, np.ndarray)
    assert np.array_equal(p_load(), p())

    p.set_backend('torch')
    p.save(path)
    p_load = LinearParameter.load(path)

    assert isinstance(p_load.initial_value, torch.Tensor)
    assert torch.equal(p_load(), p())

    p = Parameter(0.5, shape=(2, 2), log_full=True)
    path = str(tmpdir / 'param.msh')
    p.save(path)

    p2 = Parameter.load(path)

    assert p2._log_full is True
    assert p2._logger is None
    p2(0, 0)

    p = LinearParameter(1.0, threshold_value=0.0, n=10, shape=(2,), backend='torch')
    p.update(torch.tensor([0]))
    path = str(tmpdir / 'param.msh')
    p.save(path)

    p2 = Parameter.load(path)

    assert p2._backend is TorchBackend
    assert np.isclose(float(np.ravel(p2.get_value(torch.tensor([0])))[0]), 0.9)


def test_set_updates(tmpdir):
    logger = FakeLogger()
    p = LinearParameter(1., .1, n=10)
    p.set_logger(logger, 'eps')

    p()
    p.set_updates(False)
    values = [p() for _ in range(3)]

    assert np.allclose(values, [.91, .91, .91])
    assert len(logger.calls) == 1

    path = tmpdir / 'parameter.msh'
    p.save(path)
    p_load = LinearParameter.load(path)

    assert np.isclose(p_load(), .91)

    p.set_updates()
    p_load.set_updates()

    assert np.isclose(p(), .82)
    assert np.isclose(p_load(), .82)
    assert len(logger.calls) == 2

    p = LinearParameter(1., .1, n=10)
    p()
    del p._save_attributes['_updates_enabled']
    del p._updates_enabled

    path = tmpdir / 'parameter.msh'
    p.save(path)
    p_load = LinearParameter.load(path)

    assert np.isclose(p_load(), .82)

    np.random.seed(42)
    epsilon = LinearParameter(1., .1, n=10)
    pi = EpsGreedy(epsilon)
    pi.set_q(Table((10, 3)))

    epsilon.set_updates(False)
    for _ in range(3):
        pi.draw_action(np.array([2]))

    assert np.isclose(epsilon.get_value(), 1.)

    epsilon.set_updates()
    pi.draw_action(np.array([2]))

    assert np.isclose(epsilon.get_value(), .91)


def test_variance_shape():
    p = VarianceDrivenIncreasingParameter(value=1., tol=1., shape=(2,))

    p.update(0, target=1.0)
    p.update(0, target=2.0)
    p.update(0, target=1.0)

    p.update(1, target=5.0)

    assert np.isclose(p.get_value(0), 0.3333333333333333)
    assert np.isclose(p.get_value(1), 1.0)


def test_torch_index():
    p = LinearParameter(1.0, threshold_value=0.0, n=10, backend='torch')

    assert np.isclose(p(torch.tensor([3])), 0.9)

    p = LinearParameter(1.0, threshold_value=0.0, n=10, shape=(3,), backend='torch')

    p.update(torch.tensor([1]))
    p.update(torch.tensor([1]))

    assert np.isclose(float(np.ravel(p.get_value(torch.tensor([1])))[0]), 0.8)
    assert np.isclose(float(np.ravel(p.get_value(torch.tensor([0])))[0]), 1.0)
