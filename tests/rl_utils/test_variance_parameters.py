import numpy as np

from mushroom_rl.core.array_backend import TorchBackend
from mushroom_rl.rl_utils.parameters import (
    VarianceDrivenIncreasingParameter, VarianceDrivenDecreasingParameter,
    WindowedVarianceDrivenIncreasingParameter
)


def test_variance_increasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDrivenIncreasingParameter(value=1., tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01866124)


def test_variance_increasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDrivenIncreasingParameter(value=1., exponential=True, tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01515935)


def test_variance_decreasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDrivenDecreasingParameter(value=1., tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.69376147)


def test_variance_decreasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = VarianceDrivenDecreasingParameter(value=1., exponential=True, tol=1.)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.72238985)


def test_windowed_variance_increasing_linear():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = WindowedVarianceDrivenIncreasingParameter(value=1., tol=1., window=10)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.01162217)


def test_windowed_variance_increasing_exponential():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = WindowedVarianceDrivenIncreasingParameter(value=1., exponential=True, tol=1., window=10)
    for t in targets:
        p.update(0, target=t)
    assert np.isclose(p.get_value(0), 0.0095416)


def test_variance_parameter_returns_initial_before_two_updates():
    p = VarianceDrivenIncreasingParameter(value=0.5, tol=1.)
    p.update(0, target=1.)
    assert np.isclose(p.get_value(0), 0.5)


def test_variance_parameter_torch_backend_device():
    device = TorchBackend.check_device(None)
    np.random.seed(42)
    targets = np.random.randn(50).tolist()

    p = VarianceDrivenIncreasingParameter(value=1., tol=1.)
    p_torch = VarianceDrivenIncreasingParameter(value=1., tol=1., backend='torch', device=device)
    w = WindowedVarianceDrivenIncreasingParameter(value=1., tol=1., window=10)
    w_torch = WindowedVarianceDrivenIncreasingParameter(value=1., tol=1., window=10, backend='torch', device=device)
    for t in targets:
        for param in [p, p_torch, w, w_torch]:
            param.update(0, target=t)

    assert p_torch.get_value(0) == p.get_value(0)
    assert w_torch.get_value(0) == w.get_value(0)


def test_variance_parameter_set_updates():
    np.random.seed(42)
    targets = np.random.randn(50).tolist()

    p = VarianceDrivenIncreasingParameter(value=1., tol=1.)
    w = WindowedVarianceDrivenIncreasingParameter(value=1., tol=1., window=10)
    for t in targets[:25]:
        p.update(0, target=t)
        w.update(0, target=t)
    value, value_windowed = p.get_value(0), w.get_value(0)

    p.set_updates(False)
    w.set_updates(False)
    for t in targets[25:]:
        p.update(0, target=t)
        w.update(0, target=t)

    assert p.get_value(0) == value
    assert w.get_value(0) == value_windowed
