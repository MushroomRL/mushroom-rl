import numpy as np
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
    assert np.allclose(rew.mean, [[0.05275285, 0.00430447, 0.17010064]])


def test_running_exp_weighted_average_init_value():
    init = np.array([3., -1.])
    rew = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy', init_value=init)
    assert np.allclose(rew.mean, [[3., -1.]])


def test_running_exp_weighted_average_reset():
    rew = RunningExpWeightedAverage(shape=(2,), alpha=0.1, backend='numpy')
    rew.update_stats(np.array([10., 5.]))
    rew.reset()
    assert np.allclose(rew.mean, [[0., 0.]])


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