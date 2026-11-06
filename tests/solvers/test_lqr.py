from mushroom_rl.environments import LQR
import numpy as np
from mushroom_rl.solvers.lqr import (compute_lqr_feedback_gain, compute_lqr_P, compute_lqr_V, compute_lqr_Q,
                                     compute_lqr_V_gaussian_policy, compute_lqr_Q_gaussian_policy,
                                     compute_lqr_V_gaussian_policy_gradient_K, compute_lqr_Q_gaussian_policy_gradient_K)
from mushroom_rl.utils.numerical_gradient import numerical_diff_function


def test_feedback_gain():
    lqr = LQR.generate(3)
    K = compute_lqr_feedback_gain(lqr)

    K_test = np.load('tests/solvers/data/test_feedback_gain_K.npy')

    assert np.allclose(K, K_test)

    lqr = LQR.generate(3)
    K = np.array(
        [[1.0, 0.1, 0.01],
         [0.5, 1.2, 0.02],
         [.02, 0.3, 0.9]]
    )

    P = compute_lqr_P(lqr, K)

    P_test = np.load('tests/solvers/data/test_feedback_gain_P.npy')

    assert np.allclose(P, P_test)


def test_V():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    s = np.array([1.0, 1.3, -0.3])
    V_lqr = compute_lqr_V(s, lqr, K).item()

    assert np.allclose(V_lqr, -6.3336186348534875)

    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.load('tests/solvers/data/test_V_sigma.npy')

    s = np.array([1.0, 1.3, -0.3])
    V_lqg = compute_lqr_V_gaussian_policy(s, lqr, K, Sigma)

    assert np.allclose(V_lqg, -28.39165320182624)


def test_Q():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([0.5, 0.2, 0.1])

    Q_lqr = compute_lqr_Q(s, a, lqr, K).item()

    assert np.allclose(Q_lqr, -10.83964921837036)

    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.load('tests/solvers/data/test_Q_sigma.npy')

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([-0.5, -0.2, 0.1])

    Q_lqg = compute_lqr_Q_gaussian_policy(s, a, lqr, K, Sigma).item()

    assert np.allclose(Q_lqg, -23.887098201718487)

    lqr = LQR.generate(10)

    K = np.eye(10) * 0.1
    Sigma = np.eye(10) * 0.1

    s = np.ones(10)
    a = np.ones(10)
    #
    Q_lqg = compute_lqr_Q_gaussian_policy(s, a, lqr, K, Sigma).item()

    assert np.allclose(Q_lqg, -48.00590405904062)


def test_gradient_K():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.load('tests/solvers/data/test_gradient_K_sigma_0.npy')

    s = np.array([1.0, 1.3, -0.3])

    dJ = compute_lqr_V_gaussian_policy_gradient_K(s, lqr, K, Sigma)

    def compute_v(theta):
        return compute_lqr_V_gaussian_policy(s, lqr, theta.reshape(K.shape), Sigma)

    dJ_num = numerical_diff_function(compute_v, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)

    A = np.array([[1., 0.4],
                  [0.2, 0.8]])

    B = np.array([[0.8],
                  [0.5]])

    Q = np.eye(2)

    R = np.eye(1)

    lqr = LQR(A, B, Q, R, max_pos=np.inf, max_action=np.inf,
              random_init=False, episodic=False, gamma=0.9, horizon=100,
              initial_state=None)

    K = np.array([[1.0, 0.1]])

    Sigma = np.array([[0.2]])

    s = np.array([1.0, 1.3])

    dJ = compute_lqr_V_gaussian_policy_gradient_K(s, lqr, K, Sigma)

    def compute_v(theta):
        return compute_lqr_V_gaussian_policy(s, lqr, theta.reshape(K.shape), Sigma)

    dJ_num = numerical_diff_function(compute_v, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)

    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.load('tests/solvers/data/test_gradient_K_sigma_1.npy')

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([-0.6, -0.5, 0.2])

    dJ = compute_lqr_Q_gaussian_policy_gradient_K(s, a, lqr, K, Sigma)

    def compute_q(theta):
        return compute_lqr_Q_gaussian_policy(s, a, lqr, theta.reshape(K.shape), Sigma)

    dJ_num = numerical_diff_function(compute_q, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)
