import numpy as np
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer, SGDOptimizer, AdamOptimizer


def test_sgd_maximize():
    opt = SGDOptimizer(lr=0.1)
    params = np.array([0., 0., 0.])
    grads = np.array([1., -2., 0.5])
    assert np.allclose(opt(params, grads), [0.1, -0.2, 0.05])


def test_sgd_minimize():
    opt = SGDOptimizer(lr=0.1, maximize=False)
    params = np.array([0., 0., 0.])
    grads = np.array([1., -2., 0.5])
    assert np.allclose(opt(params, grads), [-0.1, 0.2, -0.05])


def test_adam_maximize():
    opt = AdamOptimizer(lr=0.01)
    p = np.zeros(3)
    g = np.array([1., -1., 0.5])
    for _ in range(20):
        p = opt(p.copy(), g.copy())
    assert np.allclose(p, [0.19999998, -0.19999998, 0.19999996])


def test_adam_minimize():
    opt = AdamOptimizer(lr=0.01, maximize=False)
    p = np.zeros(3)
    g = np.array([1., -1., 0.5])
    for _ in range(20):
        p = opt(p.copy(), g.copy())
    assert np.allclose(p, [-0.19999998, 0.19999998, -0.19999996])


def test_adaptive_with_natural_gradient():
    opt = AdaptiveOptimizer(eps=0.1)
    params = np.zeros(3)
    g = np.array([1., 0., 0.])
    nat_g = np.array([2., 0., 0.])
    assert np.allclose(opt(params, g, nat_g), [0.4472136, 0., 0.])


def test_adaptive_gradient_only():
    opt = AdaptiveOptimizer(eps=0.5)
    params = np.zeros(3)
    g = np.array([1., 1., 1.])
    assert np.allclose(opt(params, g), [0.40824829, 0.40824829, 0.40824829])


def test_adaptive_minimize():
    opt = AdaptiveOptimizer(eps=0.5, maximize=False)
    params = np.zeros(3)
    g = np.array([1., 1., 1.])
    assert np.allclose(opt(params, g), [-0.40824829, -0.40824829, -0.40824829])
