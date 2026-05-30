import numpy as np
from mushroom_rl.rl_utils.variance_parameters import (
    VarianceIncreasingParameter, VarianceDecreasingParameter,
    WindowedVarianceIncreasingParameter
)


def test_variance_increasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceIncreasingParameter(value=1., tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01866124)


def test_variance_increasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceIncreasingParameter(value=1., exponential=True, tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01515935)


def test_variance_decreasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDecreasingParameter(value=1., tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.69376147)


def test_variance_decreasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDecreasingParameter(value=1., exponential=True, tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.72238985)


def test_windowed_variance_increasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = WindowedVarianceIncreasingParameter(value=1., tol=1., window=10)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01162217)


def test_windowed_variance_increasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = WindowedVarianceIncreasingParameter(value=1., exponential=True, tol=1., window=10)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.0095416)


def test_variance_parameter_returns_initial_before_two_updates():
    p = VarianceIncreasingParameter(value=0.5, tol=1.)
    p.update(0, target=1.)
    assert np.isclose(p.get_value(0), 0.5)
