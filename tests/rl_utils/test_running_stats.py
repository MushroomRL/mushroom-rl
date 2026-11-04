import numpy as np
import pytest
import torch
from mushroom_rl.rl_utils.running_stats import (
    RunningStandardization, RunningExpWeightedAverage, RunningAveragedWindow
)


def test_running_standardization_numpy():
    np.random.seed(42)
    rs = RunningStandardization(shape=(3,), backend='numpy')
    for row in np.random.randn(20, 3):
        rs.update_stats(row)
    assert np.allclose(rs.mean, [0.05284974, -0.25307358, -0.24164668])
    assert np.allclose(rs.std, [0.74012025, 0.99835447, 0.92762235])


def test_running_standardization_torch():
    torch.manual_seed(42)
    rs = RunningStandardization(shape=(2,), backend='torch')
    for row in torch.randn(20, 2):
        rs.update_stats(row)
    assert torch.allclose(rs.mean, torch.tensor([0.1079, 0.0609]), atol=1e-4)
    assert torch.allclose(rs.std, torch.tensor([1.0341, 1.2082]), atol=1e-4)


def test_running_standardization_reset():
    rs = RunningStandardization(shape=(3,), backend='numpy')
    np.random.seed(42)
    for row in np.random.randn(20, 3):
        rs.update_stats(row)
    rs.reset()
    assert np.allclose(rs.mean, np.zeros(3))


def test_running_exp_weighted_average():
    np.random.seed(7)
    rew = RunningExpWeightedAverage(shape=(3,), alpha=0.1, backend='numpy')
    for _ in range(30):
        rew.update_stats(np.random.randn(3))
    assert np.allclose(rew.mean, [0.05275285, 0.00430447, 0.17010064])


def test_running_exp_weighted_average_init_value():
    init = np.array([3., -1.])
    rew = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy', init_value=init)
    assert np.allclose(rew.mean, [3., -1.])


def test_running_exp_weighted_average_reset():
    rew = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy')
    rew.update_stats(np.array([10., 5.]))
    rew.reset()
    assert np.allclose(rew.mean, [0., 0.])


def test_running_averaged_window():
    raw = RunningAveragedWindow(shape=(2,), window_size=5, backend='numpy')
    for i in range(10):
        raw.update_stats(np.array([float(i), float(i * 2)]))
    # window holds last 5 values: 5..9 and 10..18 step 2 → means [7, 14]
    assert np.allclose(raw.mean, [7., 14.])


def test_running_averaged_window_reset():
    raw = RunningAveragedWindow(shape=(2,), window_size=5, backend='numpy')
    for i in range(10):
        raw.update_stats(np.array([float(i), float(i * 2)]))
    raw.reset()
    assert np.allclose(raw.mean, [0., 0.])


def test_running_stats_save_the_arrays_with_the_backend_method(tmpdir):
    standardization = RunningStandardization(shape=(2,), backend='torch')
    standardization.update_stats(torch.ones(1, 2))

    average = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='torch')
    average.update_stats(torch.ones(2))

    assert standardization._save_attributes['_m'] == 'torch'
    assert standardization._save_attributes['_s'] == 'torch'
    assert average._save_attributes['_avg_value'] == 'torch'

    standardization.save(tmpdir / 'standardization.msh')
    average.save(tmpdir / 'average.msh')

    loaded_standardization = RunningStandardization.load(tmpdir / 'standardization.msh')
    loaded_average = RunningExpWeightedAverage.load(tmpdir / 'average.msh')

    assert torch.equal(loaded_standardization.mean, standardization.mean)
    assert torch.equal(loaded_standardization.std, standardization.std)
    assert torch.equal(loaded_average.mean, average.mean)

    numpy_standardization = RunningStandardization(shape=(2,), backend='numpy')

    assert numpy_standardization._save_attributes['_m'] == 'numpy'


def test_running_stats_allocate_on_the_requested_device():
    if not torch.cuda.is_available():
        pytest.skip('requires cuda')

    standardization = RunningStandardization(shape=(2,), backend='torch', device='cuda')
    standardization.update_stats(torch.ones(1, 2, device='cuda'))

    assert standardization.mean.device.type == 'cuda'
    assert standardization.std.device.type == 'cuda'

    average = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='torch', device='cuda')
    average.update_stats(torch.ones(2, device='cuda'))

    assert average.mean.device.type == 'cuda'

    initialized_average = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='torch',
                                                    init_value=np.ones(2), device='cuda')

    assert initialized_average.mean.device.type == 'cuda'


def test_running_stats_reject_a_device_on_the_numpy_backend():
    with pytest.raises(ValueError):
        RunningStandardization(shape=(2,), backend='numpy', device='cuda')

    with pytest.raises(ValueError):
        RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy', device='cuda')

    with pytest.raises(ValueError):
        RunningAveragedWindow(shape=(2,), window_size=5, backend='numpy', device='cuda')


def test_running_standardization_reset_matches_a_fresh_instance():
    rs = RunningStandardization(shape=(3,), backend='numpy')
    fresh = RunningStandardization(shape=(3,), backend='numpy')
    np.random.seed(42)
    data = np.random.randn(20, 3)

    for row in data:
        rs.update_stats(row)
    rs.reset()

    assert rs._m.shape == fresh._m.shape
    assert rs._s.shape == fresh._s.shape

    for row in data:
        rs.update_stats(row)
        fresh.update_stats(row)

    assert rs._m.shape == fresh._m.shape
    assert np.allclose(rs.mean, fresh.mean)
    assert np.allclose(rs.std, fresh.std)


def test_running_standardization_to_keeps_the_statistics():
    standardization = RunningStandardization(shape=(2,), backend='numpy')
    standardization.update_stats(np.array([[1., 2.], [3., 5.]]))
    mean, std = standardization.mean.copy(), standardization.std.copy()

    assert standardization.to(None) is standardization
    assert standardization.device is None
    assert np.array_equal(standardization.mean, mean)
    assert np.array_equal(standardization.std, std)


def test_running_standardization_to_moves_the_statistics():
    if not torch.cuda.is_available():
        pytest.skip('requires cuda')

    standardization = RunningStandardization(shape=(2,), backend='torch', device='cpu')
    standardization.update_stats(torch.tensor([[1., 2.], [3., 5.]]))
    m, s = standardization._m.clone(), standardization._s.clone()

    standardization.to('cuda')

    assert standardization.device == 'cuda'
    assert standardization.mean.device.type == 'cuda'
    assert standardization.std.device.type == 'cuda'
    assert torch.equal(standardization._m.cpu(), m)
    assert torch.equal(standardization._s.cpu(), s)


def test_running_stats_read_a_multidimensional_value_as_a_single_sample():
    standardization = RunningStandardization(shape=(2, 3), backend='numpy')
    average = RunningExpWeightedAverage(shape=(2, 3), alpha=0.5, backend='numpy')
    window = RunningAveragedWindow(shape=(2, 3), window_size=5, backend='numpy')
    samples = np.arange(12.).reshape(2, 2, 3)

    for sample in samples:
        for statistic in (standardization, average, window):
            statistic.update_stats(sample)

    assert standardization._n == 3
    assert np.allclose(standardization.mean, [[2., 2.6666667, 3.3333333], [4., 4.6666667, 5.3333333]])
    assert np.allclose(standardization.std, [[2.8867513, 3.1446604, 3.4480268], [3.7859389, 4.1499665, 4.5338235]])
    assert np.allclose(average.mean, [[3., 3.75, 4.5], [5.25, 6., 6.75]])
    assert len(window._avg_buffer) == 3
    assert np.allclose(window.mean, [[2., 2.6666667, 3.3333333], [4., 4.6666667, 5.3333333]])


def test_running_stats_read_a_batch_like_the_same_single_samples():
    np.random.seed(3)
    samples = np.random.randn(4, 2, 3)

    for make in (lambda: RunningStandardization(shape=(2, 3), backend='numpy'),
                 lambda: RunningExpWeightedAverage(shape=(2, 3), alpha=0.5, backend='numpy'),
                 lambda: RunningAveragedWindow(shape=(2, 3), window_size=3, backend='numpy')):
        batched, sequential = make(), make()
        batched.update_stats(samples)
        for sample in samples:
            sequential.update_stats(sample)

        assert np.allclose(batched.mean, sequential.mean)


def test_running_stats_mean_has_the_sample_shape():
    for shape in [(3,), (2, 3)]:
        standardization = RunningStandardization(shape=shape, backend='numpy')
        average = RunningExpWeightedAverage(shape=shape, alpha=0.1, backend='numpy')
        initialized_average = RunningExpWeightedAverage(shape=shape, alpha=0.1, backend='numpy',
                                                        init_value=np.ones(shape))
        window = RunningAveragedWindow(shape=shape, window_size=5, backend='numpy')

        for statistic in (standardization, average, initialized_average, window):
            assert statistic.mean.shape == shape
            statistic.update_stats(np.ones(shape))
            assert statistic.mean.shape == shape


def test_running_averaged_window_mean_on_the_list_backend():
    window = RunningAveragedWindow(shape=(2,), window_size=3, backend='list')
    for i in range(4):
        window.update_stats([float(i), float(i * 2)])

    assert np.allclose(window.mean, [2., 4.])


def test_running_averaged_window_mean_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip('requires cuda')

    window = RunningAveragedWindow(shape=(2,), window_size=3, backend='torch', device='cuda')
    for i in range(4):
        window.update_stats(torch.tensor([float(i), float(i * 2)], device='cuda'))

    assert window.mean.device.type == 'cuda'
    assert torch.allclose(window.mean.cpu(), torch.tensor([2., 4.]))


def test_running_standardization_keeps_the_unit_dimensions_of_the_sample():
    for shape in [(1,), (3, 1)]:
        standardization = RunningStandardization(shape=shape, backend='numpy')
        standardization.update_stats(np.ones(shape))

        assert standardization.mean.shape == shape
        assert standardization.std.shape == shape
