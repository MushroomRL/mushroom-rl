from mushroom_rl.policy.td_policy import *
from mushroom_rl.approximators.table import Table
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter


def test_td_policy():
    Q = Table((10, 3))
    pi = TDPolicy()

    pi.set_q(Q)

    assert Q == pi.get_q()


def test_eps_greedy():
    np.random.seed(88)
    eps = Parameter(0.1)
    pi = EpsGreedy(eps)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([1])

    p_s = pi(s)
    p_s_test = np.array([0.03333333, 0.93333333, 0.03333333])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.93333333])
    assert np.allclose(p_sa, p_sa_test)

    a, _ = pi.draw_action(s)
    a_test = 1
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


def test_boltzmann():
    np.random.seed(88)
    beta = Parameter(0.1)
    pi = Boltzmann(beta)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([1])

    p_s = pi(s)
    p_s_test = np.array([0.30676679, 0.36223227, 0.33100094])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.36223227])
    assert np.allclose(p_sa, p_sa_test)

    a, _ = pi.draw_action(s)
    a_test = 2
    assert a.item() == a_test

    beta_2 = LinearParameter(0.2, 0.1, 2)
    pi.set_beta(beta_2)
    p_sa_2 = pi(s, a)
    assert p_sa_2 < p_sa

    pi.update(s, a)
    p_sa_3 = pi(s, a)
    p_sa_3_test = np.array([0.33100094])
    assert np.allclose(p_sa_3, p_sa_3_test)


def test_mellowmax():
    np.random.seed(88)
    omega = Parameter(3)
    pi = Mellowmax(omega)

    Q = Table((10, 3))
    Q.table = np.random.randn(10, 3)

    pi.set_q(Q)

    s = np.array([2])
    a = np.array([1])

    p_s = pi(s)
    p_s_test = np.array([0.08540336, 0.69215916, 0.22243748])
    assert np.allclose(p_s, p_s_test)

    p_sa = pi(s, a)
    p_sa_test = np.array([0.69215916])
    assert np.allclose(p_sa, p_sa_test)

    a, _ = pi.draw_action(s)
    a_test = 2
    assert a.item() == a_test

    try:
        beta = Parameter(0.1)
        pi.set_beta(beta)
    except RuntimeError:
        pass
    else:
        assert False

    try:
        pi.update(s,a)
    except RuntimeError:
        pass
    else:
        assert False
