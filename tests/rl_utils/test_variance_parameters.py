import numpy as np
import pytest

from mushroom_rl.core.array_backend import TorchBackend
from mushroom_rl.rl_utils.parameters import (
    VarianceDrivenIncreasingParameter, VarianceDrivenDecreasingParameter,
    WindowedVarianceDrivenIncreasingParameter
)


@pytest.mark.parametrize('parameter, kwargs, value', [
    (VarianceDrivenIncreasingParameter, dict(), 0.01866124),
    (VarianceDrivenIncreasingParameter, dict(exponential=True), 0.01515935),
    (VarianceDrivenDecreasingParameter, dict(), 0.69376147),
    (VarianceDrivenDecreasingParameter, dict(exponential=True), 0.72238985),
    (WindowedVarianceDrivenIncreasingParameter, dict(window=10), 0.01162217),
    (WindowedVarianceDrivenIncreasingParameter, dict(exponential=True, window=10), 0.0095416)
])
def test_variance(parameter, kwargs, value):
    np.random.seed(42)
    targets = np.random.randn(50).tolist()
    p = parameter(value=1., tol=1., **kwargs)
    for t in targets:
        p.update(0, target=t)

    assert np.isclose(p.get_value(0), value)


def test_variance_backend():
    p = VarianceDrivenIncreasingParameter(value=0.5, tol=1.)
    p.update(0, target=1.)

    assert np.isclose(p.get_value(0), 0.5)

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


def test_variance_set_updates():
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
