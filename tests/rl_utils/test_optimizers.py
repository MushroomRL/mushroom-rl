import numpy as np
import pytest

from mushroom_rl.rl_utils.optimizers import Optimizer, AdaptiveOptimizer, SGDOptimizer, AdamOptimizer
from mushroom_rl.rl_utils.parameters import Parameter


def test_sgd():
    opt = SGDOptimizer(lr=0.1)
    params = np.array([0., 0., 0.])
    grads = np.array([1., -2., 0.5])
    assert np.allclose(opt(params, grads), [0.1, -0.2, 0.05])

    opt = SGDOptimizer(lr=0.1, maximize=False)
    params = np.array([0., 0., 0.])
    grads = np.array([1., -2., 0.5])
    assert np.allclose(opt(params, grads), [-0.1, 0.2, -0.05])

    assert np.allclose(SGDOptimizer(lr=Parameter(0.1))(np.zeros(3), np.array([1., -2., 0.5])), [0.1, -0.2, 0.05])

    with pytest.raises(NotImplementedError):
        Optimizer()(params, grads)


def test_adam():
    opt = AdamOptimizer(lr=0.01)
    p = np.zeros(3)
    g = np.array([1., -1., 0.5])
    for _ in range(20):
        p = opt(p.copy(), g)
    assert np.allclose(p, [0.19999998, -0.19999998, 0.19999996])

    opt = AdamOptimizer(lr=0.01, maximize=False)
    p = np.zeros(3)
    g = np.array([1., -1., 0.5])
    for _ in range(20):
        p = opt(p.copy(), g)
    assert np.allclose(p, [-0.19999998, 0.19999998, -0.19999996])


def test_adaptive():
    opt = AdaptiveOptimizer(eps=0.1)
    params = np.zeros(3)
    g = np.array([1., 0., 0.])
    nat_g = np.array([2., 0., 0.])
    assert np.allclose(opt(params, g, nat_g), [0.4472136, 0., 0.])

    opt = AdaptiveOptimizer(eps=0.5)
    params = np.zeros(3)
    g = np.array([1., 1., 1.])
    assert np.allclose(opt(params, g), [0.40824829, 0.40824829, 0.40824829])

    opt = AdaptiveOptimizer(eps=0.5, maximize=False)
    params = np.zeros(3)
    g = np.array([1., 1., 1.])
    assert np.allclose(opt(params, g), [-0.40824829, -0.40824829, -0.40824829])

    with pytest.raises(ValueError):
        opt.get_value(g, g, g)


def test_no_mutation():
    for opt in (SGDOptimizer(lr=0.1, maximize=False), AdamOptimizer(lr=0.1, maximize=False),
                AdaptiveOptimizer(eps=0.1, maximize=False)):
        g = np.array([1., -2., 0.5])
        opt(np.zeros(3), g)

        assert np.array_equal(g, [1., -2., 0.5])

    g, nat_g = np.array([1., 0., 0.]), np.array([2., 0., 0.])

    assert np.allclose(AdaptiveOptimizer(eps=0.1, maximize=False)(np.zeros(3), g, nat_g), [-0.4472136, 0., 0.])
    assert np.array_equal(g, [1., 0., 0.]) and np.array_equal(nat_g, [2., 0., 0.])
