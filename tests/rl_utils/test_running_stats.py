import numpy as np
import pytest
import torch

from helper.arrays import assert_backend, make_array, to_numpy
from helper.cases import backend_devices, devices

from mushroom_rl.rl_utils.running_stats import (
    RunningStandardization, RunningExpWeightedAverage, RunningAveragedWindow
)


def make_statistics(shape, backend, device, window_size=5, alpha=0.5):
    return (RunningStandardization(shape=shape, backend=backend, device=device),
            RunningExpWeightedAverage(shape=shape, alpha=alpha, backend=backend, device=device),
            RunningAveragedWindow(shape=shape, window_size=window_size, backend=backend, device=device))


def test_standardization():
    np.random.seed(42)
    data = np.random.randn(20, 3)
    rs = RunningStandardization(shape=(3,), backend='numpy')
    fresh = RunningStandardization(shape=(3,), backend='numpy')
    for row in data:
        rs.update_stats(row)

    assert np.allclose(rs.mean, [0.05284974, -0.25307358, -0.24164668])
    assert np.allclose(rs.std, [0.74012025, 0.99835447, 0.92762235])

    rs.reset()

    assert np.allclose(rs.mean, np.zeros(3))
    assert rs._m.shape == fresh._m.shape
    assert rs._s.shape == fresh._s.shape

    for row in data:
        rs.update_stats(row)
        fresh.update_stats(row)

    assert rs._m.shape == fresh._m.shape
    assert np.allclose(rs.mean, fresh.mean)
    assert np.allclose(rs.std, fresh.std)

    torch.manual_seed(42)
    rs = RunningStandardization(shape=(2,), backend='torch')
    for row in torch.randn(20, 2):
        rs.update_stats(row)

    assert torch.allclose(rs.mean, torch.tensor([0.1079, 0.0609]), atol=1e-4)
    assert torch.allclose(rs.std, torch.tensor([1.0341, 1.2082]), atol=1e-4)


def test_exp_weighted_average():
    np.random.seed(7)
    rew = RunningExpWeightedAverage(shape=(3,), alpha=0.1, backend='numpy')
    for _ in range(30):
        rew.update_stats(np.random.randn(3))

    assert np.allclose(rew.mean, [0.05275285, 0.00430447, 0.17010064])

    rew.reset()

    assert np.allclose(rew.mean, [0., 0., 0.])
    assert np.allclose(RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy',
                                                 init_value=np.array([3., -1.])).mean, [3., -1.])


@pytest.mark.parametrize('backend, device', backend_devices())
def test_averaged_window(backend, device):
    window = RunningAveragedWindow(shape=(2,), window_size=5, backend=backend, device=device)
    for i in range(10):
        window.update_stats(make_array(backend, [float(i), float(i * 2)], device))

    assert np.allclose(to_numpy(window.mean), [7., 14.])

    window.reset()

    assert np.allclose(to_numpy(window.mean), [0., 0.])

    window = RunningAveragedWindow(shape=(2,), window_size=3, backend=backend, device=device)
    for i in range(4):
        window.update_stats(make_array(backend, [float(i), float(i * 2)], device))

    assert np.allclose(to_numpy(window.mean), [2., 4.])
    if backend == 'torch':
        assert window.mean.device.type == device


def test_save(tmpdir):
    standardization = RunningStandardization(shape=(2,), backend='torch')
    standardization.update_stats(torch.ones(1, 2))

    average = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='torch')
    average.update_stats(torch.ones(2))

    assert standardization._save_attributes['_m'] == 'torch'
    assert standardization._save_attributes['_s'] == 'torch'
    assert average._save_attributes['_avg_value'] == 'torch'
    assert RunningStandardization(shape=(2,), backend='numpy')._save_attributes['_m'] == 'numpy'

    standardization.save(tmpdir / 'standardization.msh')
    average.save(tmpdir / 'average.msh')

    loaded_standardization = RunningStandardization.load(tmpdir / 'standardization.msh')
    loaded_average = RunningExpWeightedAverage.load(tmpdir / 'average.msh')

    assert torch.equal(loaded_standardization.mean, standardization.mean)
    assert torch.equal(loaded_standardization.std, standardization.std)
    assert torch.equal(loaded_average.mean, average.mean)


@pytest.mark.parametrize('device', devices())
def test_device(device):
    standardization, average, _ = make_statistics((2,), 'torch', device, alpha=0.1)
    standardization.update_stats(torch.ones(1, 2, device=device))
    average.update_stats(torch.ones(2, device=device))
    initialized_average = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='torch', init_value=np.ones(2),
                                                    device=device)

    assert standardization.mean.device.type == device
    assert standardization.std.device.type == device
    assert average.mean.device.type == device
    assert initialized_average.mean.device.type == device

    standardization = RunningStandardization(shape=(2,), backend='torch', device='cpu')
    standardization.update_stats(torch.tensor([[1., 2.], [3., 5.]]))
    m, s = standardization._m.clone(), standardization._s.clone()

    standardization.to(device)

    assert standardization.device == device
    assert standardization.mean.device.type == device
    assert standardization.std.device.type == device
    assert torch.equal(standardization._m.cpu(), m)
    assert torch.equal(standardization._s.cpu(), s)

    standardization = RunningStandardization(shape=(2,), backend='numpy')
    standardization.update_stats(np.array([[1., 2.], [3., 5.]]))
    mean, std = standardization.mean.copy(), standardization.std.copy()

    assert standardization.to(None) is standardization
    assert standardization.device is None
    assert np.array_equal(standardization.mean, mean)
    assert np.array_equal(standardization.std, std)

    with pytest.raises(ValueError):
        RunningStandardization(shape=(2,), backend='numpy', device=device)
    with pytest.raises(ValueError):
        RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy', device=device)
    with pytest.raises(ValueError):
        RunningAveragedWindow(shape=(2,), window_size=5, backend='numpy', device=device)


@pytest.mark.parametrize('backend, device', backend_devices('numpy', 'torch'))
def test_sample_shape(backend, device):
    standardization, average, window = make_statistics((2, 3), backend, device)
    for sample in np.arange(12.).reshape(2, 2, 3):
        for statistic in (standardization, average, window):
            statistic.update_stats(make_array(backend, sample, device))

    assert standardization._n == 3
    assert np.allclose(to_numpy(standardization.mean), np.load('tests/rl_utils/data/test_sample_shape_mean.npy'))
    assert np.allclose(to_numpy(standardization.std), np.load('tests/rl_utils/data/test_sample_shape_std.npy'))
    assert np.allclose(to_numpy(average.mean), [[3., 3.75, 4.5], [5.25, 6., 6.75]])
    assert len(window._avg_buffer) == 3
    assert np.allclose(to_numpy(window.mean), np.load('tests/rl_utils/data/test_sample_shape_window.npy'))

    np.random.seed(3)
    samples = np.random.randn(4, 2, 3)
    for batched, sequential in zip(make_statistics((2, 3), backend, device, window_size=3),
                                   make_statistics((2, 3), backend, device, window_size=3)):
        batched.update_stats(make_array(backend, samples, device))
        for sample in samples:
            sequential.update_stats(make_array(backend, sample, device))

        assert np.allclose(to_numpy(batched.mean), to_numpy(sequential.mean))

    for shape in [(3,), (2, 3), (1,), (3, 1)]:
        initialized_average = RunningExpWeightedAverage(shape=shape, alpha=0.1, backend=backend,
                                                        init_value=np.ones(shape), device=device)
        for statistic in make_statistics(shape, backend, device, alpha=0.1) + (initialized_average,):
            assert tuple(statistic.mean.shape) == shape
            statistic.update_stats(make_array(backend, np.ones(shape), device))
            assert tuple(statistic.mean.shape) == shape
            assert_backend(statistic.mean, backend, device)

        standardization = make_statistics(shape, backend, device)[0]
        standardization.update_stats(make_array(backend, np.ones(shape), device))

        assert tuple(standardization.std.shape) == shape
