from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import *
from mushroom_rl.utils.numerical_gradient import numerical_diff_function



def test_lqr_solver_linear():
    lqr = LQR.generate(3)
    K = compute_lqr_feedback_gain(lqr)

    K_test = np.array([[0.89908343, 0., 0.],
                             [0., 0.24025307, 0.],
                             [0., 0., 0.24025307]])

    assert np.allclose(K, K_test)


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

    s = np.array([1.0, 1.3, -0.3])
    V_lqr = compute_lqr_V(s, lqr, K).item()

    assert np.allclose(V_lqr, -6.3336186348534875)


def test_V_lqr_gaussian_policy():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, 0.19607835],
                      [0.02205161,  0.59897771,  0.09953863],
                      [0.19607835,  0.09953863,  0.23284475]])

    s = np.array([1.0, 1.3, -0.3])
    V_lqg = compute_lqr_V_gaussian_policy(s, lqr, K, Sigma)
    
    assert np.allclose(V_lqg, -28.39165320182624)


def test_Q_lqr():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([0.5, 0.2, 0.1])

    Q_lqr = compute_lqr_Q(s, a, lqr, K).item()

    assert np.allclose(Q_lqr, -10.83964921837036)


def test_Q_lqr_gaussian_policy():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, 0.19607835],
                      [0.02205161,  0.59897771, 0.09953863],
                      [0.19607835, 0.09953863, 0.23284475]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([-0.5, -0.2, 0.1])

    Q_lqg = compute_lqr_Q_gaussian_policy(s, a, lqr, K, Sigma).item()

    assert np.allclose(Q_lqg, -23.887098201718487)


def test_Q_lqr_gaussian_policy_10dim():
    lqr = LQR.generate(10)

    K = np.eye(10) * 0.1
    Sigma = np.eye(10) * 0.1

    s = np.ones(10)
    a = np.ones(10)
    #
    Q_lqg = compute_lqr_Q_gaussian_policy(s, a, lqr, K, Sigma).item()

    assert np.allclose(Q_lqg, -48.00590405904062)


def test_V_lqr_gaussian_policy_gradient_K():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, -0.19607835],
                      [0.02205161,  0.59897771,  0.09953863],
                      [-0.19607835,  0.09953863,  0.23284475]])

    s = np.array([1.0, 1.3, -0.3])

    dJ = compute_lqr_V_gaussian_policy_gradient_K(s, lqr, K, Sigma)

    f = lambda theta: compute_lqr_V_gaussian_policy(s, lqr, theta.reshape(K.shape), Sigma)
    dJ_num = numerical_diff_function(f, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)


def test_V_lqr_gaussian_policy_gradient_K_diff_dims():

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

    f = lambda theta: compute_lqr_V_gaussian_policy(s, lqr, theta.reshape(K.shape), Sigma)
    dJ_num = numerical_diff_function(f, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)


def test_Q_lqr_gaussian_policy_gradient_K():
    lqr = LQR.generate(3)

    K = np.array([[1.0, 0.1, 0.01],
                  [0.5, 1.2, 0.02],
                  [.02, 0.3, 0.9]])

    Sigma = np.array([[0.18784063,  0.02205161, -0.19607835],
                      [0.02205161,  0.59897771,  0.09953863],
                      [-0.19607835,  0.09953863,  0.23284475]])

    s = np.array([1.0, 1.3, -0.3])
    a = np.array([-0.6, -0.5, 0.2])

    dJ = compute_lqr_Q_gaussian_policy_gradient_K(s, a, lqr, K, Sigma)

    f = lambda theta: compute_lqr_Q_gaussian_policy(s, a, lqr, theta.reshape(K.shape), Sigma)
    dJ_num = numerical_diff_function(f, K.reshape(-1))

    assert np.allclose(dJ, dJ_num)
