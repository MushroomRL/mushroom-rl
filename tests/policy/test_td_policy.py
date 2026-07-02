from mushroom_rl.policy.td_policy import *
from mushroom_rl.approximators.table import Table
from mushroom_rl.approximators import QApproximator
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter


def test_td_policy():
    Q = Table((10, 3))
    pi = TDPolicy()

    pi.set_q(Q)

    assert Q == pi.get_q()


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
    print(eps_2.get_value())
    assert p_sa_3 == p_sa


def test_eps_greedy_torch():
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
    p_sa_3_test = np.array([0.33690263])
    assert np.allclose(p_sa_3, p_sa_3_test)


def test_boltzmann_torch():
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

    try:
        beta = Parameter(0.1)
        pi.set_beta(beta)
    except RuntimeError:
        pass
    else:
        assert False

    try:
        pi.update(s, a)
    except RuntimeError:
        pass
    else:
        assert False
