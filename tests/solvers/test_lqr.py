import numpy as np

from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import solve_lqr_linear, compute_lqr_P, compute_lqr_V, compute_lqg_V, compute_lqg_gradient
from mushroom_rl.solvers.lqr import compute_lqr_Q, compute_lqg_Q, _parse_lqr
from mushroom_rl.utils.numerical_gradient import numerical_diff_function


def test_lqr_solver_linear():
    lqr = LQR.generate(3)
    k = solve_lqr_linear(lqr)

    k_test = np.array(
        [[0.89908343, 0., 0.],
         [0., 0.24025307, 0.],
         [0., 0., 0.24025307]]
    )

    assert np.allclose(k, k_test)


def test_P():
    lqr = LQR.generate(3)
    K = np.array(
        [[1.0, 0.1, 0.01],
         [0.5, 1.2, 0.02],
         [.02, 0.3, 0.9]]
    )

    P = compute_lqr_P(lqr, K)

    P_test = np.array([[1.60755632, 0.78058807, 0.03219049],
                       [0.78058807, 1.67738666, 0.24905620],
                       [0.03219049, 0.2490562 , 0.83697781]])

    assert np.allclose(P, P_test)


def test_V_lqr():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    x = np.array([1.0, 1.3, -0.3])
    V_lqr = compute_lqr_V(x, lqr, K)

    assert V_lqr == -6.3336186348534875


def test_V_lqg():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, 0.19607835],
                      [0.02205161,  0.59897771,  0.09953863],
                      [0.19607835,  0.09953863,  0.23284475]])

    x = np.array([1.0, 1.3, -0.3])
    V_lqg = compute_lqg_V(x, lqr, K, Sigma)

    assert V_lqg == -28.39165320182624


def test_Q_lqr():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([0.5, 0.2, 0.1])
    x = np.hstack((s, a)).reshape((-1, 1))

    Q_lqr = compute_lqr_Q(x, lqr, K).item()

    assert Q_lqr == -10.83964921837036


def test_Q_lqg():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, 0.19607835],
                      [0.02205161,  0.59897771, 0.09953863],
                      [0.19607835, 0.09953863, 0.23284475]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([0.5, 0.2, 0.1])
    x = np.hstack((s, a)).reshape((-1, 1))

    Q_lqg = compute_lqg_Q(x, lqr, K, Sigma).item()

    assert Q_lqg == -30.691880328645837


def test_V_gradient_lqg():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, -0.19607835],
                      [0.02205161,  0.59897771,  0.09953863],
                      [-0.19607835,  0.09953863,  0.23284475]])

    x = np.array([1.0, 1.3, -0.3])

    dJ = compute_lqg_gradient(x, lqr, K, Sigma)

    f = lambda theta: compute_lqg_V(x, lqr, theta.reshape(K.shape), Sigma)
    dJ_num = numerical_diff_function(f, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)
