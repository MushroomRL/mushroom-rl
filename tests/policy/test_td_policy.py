import torch

import numpy as np

import pytest

from helper.arrays import assert_backend, assert_equal, make_array, to_numpy
from helper.cases import backend_devices
from helper.dummies import FakeLogger, FixedQ

from mushroom_rl.policy.td_policy import EpsGreedy, Boltzmann, Mellowmax
from mushroom_rl.approximators.table import Table
from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter


def test_eps_greedy():
    np.random.seed(42)
    eps = Parameter(0.1)
    pi = EpsGreedy(eps)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([0])

    p_s = pi(s)
    p_s_test = np.array([0.93333333, 0.03333333, 0.03333333])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.93333333])
    assert np.allclose(p_sa, p_sa_test)

    a = pi.draw_action(s)
    a_test = 0
    assert a.item() == a_test

    eps_2 = LinearParameter(0.2, 0.1, 2)
    pi.set_epsilon(eps_2)
    p_sa_2 = pi(s, a)
    assert p_sa_2 < p_sa

    pi.update(s, a)
    pi.update(s, a)
    p_sa_3 = pi(s, a)
    assert p_sa_3 == p_sa

    np.random.seed(42)
    torch.manual_seed(42)
    eps = Parameter(0.1)
    pi = EpsGreedy(eps, backend='torch')

    Q = QApproximator(TorchApproximator, n_actions=3, output_shape=(3,), input_shape=(4,), network=LinearNetwork)
    pi.set_q(Q)

    s = torch.rand(4)

    p_s = pi(s)
    p_s_test = np.array([0.93333333, 0.03333333, 0.03333333])
    assert np.allclose(p_s, p_s_test)

    a = pi.draw_action(s)
    assert isinstance(a, torch.Tensor)
    assert a.item() == 0

    np.random.seed(42)
    eps = Parameter(0.1)
    pi = EpsGreedy(eps)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])

    a = pi.draw_action_greedy(s)
    assert a.item() == 0

    for _ in range(5):
        assert pi.draw_action_greedy(s).item() == 0


def test_boltzmann():
    np.random.seed(42)
    beta = Parameter(0.1)
    pi = Boltzmann(beta)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([0])

    p_s = pi(s)
    p_s_test = np.array([0.36539237, 0.33690263, 0.297705])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.36539237])
    assert np.allclose(p_sa, p_sa_test)

    a = pi.draw_action(s)
    a_test = 1
    assert a.item() == a_test

    beta_2 = LinearParameter(0.2, 0.1, 2)
    pi.set_beta(beta_2)
    p_sa_2 = pi(s, a)
    assert p_sa_2 < p_sa

    pi.update(s, a)
    p_sa_3 = pi(s, a)
    p_sa_3_test = np.array([0.33782081])
    assert np.allclose(p_sa_3, p_sa_3_test)

    assert beta_2._n_updates.table.item() == 1

    np.random.seed(42)
    torch.manual_seed(42)
    beta = Parameter(0.5)
    pi = Boltzmann(beta, backend='torch')

    Q = QApproximator(TorchApproximator, n_actions=3, output_shape=(3,), input_shape=(4,), network=LinearNetwork)
    pi.set_q(Q)

    s = torch.rand(4)

    p_s = pi(s)
    p_s_test = np.array([0.36522284, 0.32458198, 0.31019512])
    assert np.allclose(p_s, p_s_test)

    a = pi.draw_action(s)
    assert isinstance(a, torch.Tensor)
    assert a.item() == 2

    np.random.seed(42)
    beta = Parameter(0.1)
    pi = Boltzmann(beta)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])

    a = pi.draw_action_greedy(s)
    assert a.item() == 0

    for _ in range(5):
        assert pi.draw_action_greedy(s).item() == 0

    np.random.seed(42)
    beta = LinearParameter(value=5., threshold_value=1., n=100)
    pi = Boltzmann(beta)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])

    pi(s)
    pi(s, np.array([0]))
    pi.draw_action_greedy(s)

    assert beta.get_value() == 5.

    for _ in range(10):
        pi.draw_action(s)

    assert np.allclose(beta.get_value(), 4.6)


def test_mellowmax():
    np.random.seed(42)
    omega = Parameter(3)
    pi = Mellowmax(omega)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([1])

    p_s = pi(s)
    p_s_test = np.array([0.67744364, 0.26133716, 0.0612192])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.26133716])
    assert np.allclose(p_sa, p_sa_test)

    a = pi.draw_action(s)
    a_test = 1
    assert a.item() == a_test

    pi = Mellowmax(Parameter(2.5))

    Q = Table((1, 4))
    Q.table = np.array([[0., 0., 0., 2e-11]])

    pi.set_q(Q)

    s = np.array([0])

    p_s = pi(s)
    p_s_test = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(p_s, p_s_test)

    np.random.seed(42)
    pi = Mellowmax(Parameter(1.), beta_min=1., beta_max=2.)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])

    p_s = pi(s)
    p_s_test = np.array([0.63573931, 0.28231134, 0.08194935])
    assert np.allclose(p_s, p_s_test)

    pi = Mellowmax(Parameter(2.5))

    Q = Table((1, 4))

    pi.set_q(Q)

    s = np.array([0])

    p_s = pi(s)
    p_s_test = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(p_s, p_s_test)

    np.random.seed(42)
    pi = Mellowmax(Parameter(3), beta_max=1.)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])

    p_s = pi(s)
    p_s_test = np.array([0.63573931, 0.28231134, 0.08194935])
    assert np.allclose(p_s, p_s_test)

    pi = Mellowmax(Parameter(1.))
    pi.set_q(FixedQ(np.array([[1., 3., 2.], [.3, -.2, .1]], dtype=np.float32)))

    s = np.array([[0], [1]])

    p_s = pi(s)
    assert p_s.dtype == np.float32
    for i in range(2):
        assert np.array_equal(p_s[i], pi(s[i]))


@pytest.mark.parametrize('backend, device', backend_devices('numpy', 'torch'))
def test_td_policy_batch(backend, device):
    np.random.seed(1)
    torch.manual_seed(1)
    kw = dict(backend=backend) if backend == 'numpy' else dict(backend=backend, device=device)
    case = device or backend
    q = make_array(backend, [[1., 3., 2.], [4., 4., 0.], [0., 0., 0.], [-1., 2., -3.]], device)
    state = make_array(backend, [[0], [1], [2], [3]], device)
    action = make_array(backend, [[1], [0], [2], [0]], device)

    eps = EpsGreedy(Parameter(.5), **kw)
    eps.set_q(FixedQ(q))

    greedy = dict(numpy=[[1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], cpu=[[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1]],
                  cuda=[[1, 1, 2, 1], [1, 1, 2, 1], [1, 0, 2, 1]])[case]
    for expected in greedy:
        a = eps.draw_action_greedy(state)
        assert_backend(a, backend, device)
        assert a.shape == (4, 1)
        assert_equal(a[:, 0], expected)

    draws = dict(numpy=[[1, 1, 1, 1], [2, 1, 2, 1], [1, 2, 0, 1]], cpu=[[1, 1, 0, 1], [1, 0, 1, 1], [2, 1, 0, 1]],
                 cuda=[[0, 0, 1, 1], [0, 2, 1, 2], [2, 0, 1, 2]])[case]
    for expected in draws:
        assert_equal(eps.draw_action(state)[:, 0], expected)

    probs = eps(state)
    assert probs.shape == (4, 3)
    assert np.allclose(to_numpy(probs), [[1 / 6, 2 / 3, 1 / 6], [5 / 12, 5 / 12, 1 / 6], [1 / 3] * 3,
                                         [1 / 6, 2 / 3, 1 / 6]])
    assert np.allclose(to_numpy(eps(state, action)), [2 / 3, 5 / 12, 1 / 3, 1 / 6])
    for i in range(4):
        assert_equal(eps(state, action)[i], eps(state[i], action[i]))
    assert eps(state[0], action[0]).shape == ()
    assert eps.draw_action(state[:1]).shape == (1, 1)

    np.random.seed(2)
    torch.manual_seed(2)
    boltzmann = Boltzmann(Parameter(1.), **kw)
    boltzmann.set_q(FixedQ(q))

    draws = dict(numpy=[[1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1]], cpu=[[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]],
                 cuda=[[1, 0, 2, 1], [1, 0, 0, 1], [2, 0, 2, 1]])[case]
    for expected in draws:
        assert_equal(boltzmann.draw_action(state)[:, 0], expected)

    probs = boltzmann(state)
    for i in range(4):
        assert_equal(probs[i], boltzmann(state[i]))
        assert_equal(boltzmann(state, action)[i], boltzmann(state[i], action[i]))
    assert boltzmann(state[0], action[0]).shape == ()

    np.random.seed(3)
    torch.manual_seed(3)
    mellowmax = Mellowmax(Parameter(1.), **kw)
    mellowmax.set_q(FixedQ(q))
    logger = FakeLogger()
    mellowmax.set_logger(logger)

    draws = dict(numpy=[[1, 1, 0, 1], [2, 1, 0, 1], [0, 0, 0, 1]], cpu=[[0, 0, 0, 1], [1, 0, 1, 1], [1, 1, 2, 1]],
                 cuda=[[0, 1, 0, 1], [1, 1, 1, 2], [2, 2, 2, 1]])[case]
    for expected in draws:
        assert_equal(mellowmax.draw_action(state)[:, 0], expected)
    assert logger.calls == [{'policy/omega': 1.}] * 3

    probs = mellowmax(state)
    for i in range(4):
        assert_equal(probs[i], mellowmax(state[i]))
