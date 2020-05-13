import numpy as np
from sklearn.metrics import pairwise_distances

"""
Collection of functions to compute the optimal Policy, the state Value Function V, the state-action Value Function Q,
the gradient of V and Q, ..., of: 
    - Infinite horizon, dicounted, discrete-time LQR. a=-Ks
    - Infinite horizon, dicounted, discrete-time LQG. a=Gaussian(-Ks, Sigma)

Note that all functions are derived with a=-Ks (instead of a=Ks).

"""

def solve_lqr_linear(lqr, max_iterations=100):
    """
    Computes the optimal controller K.

    Args:
        lqr (LQR): LQR environment
        max_iterations (int): max iterations for convergence

    Returns:
        The feedback gain matrix K

    """
    A, B, Q, R, gamma = _parse_lqr(lqr)

    P = np.eye(Q.shape[0])
    K = _compute_riccati_gain(P, A, B, R, gamma)

    it = 0
    while it < max_iterations:
        P = _compute_riccati_rhs(A, B, Q, R, gamma, K, P)
        K = _compute_riccati_gain(P, A, B, R, gamma)

        it += 1

    return K


def compute_lqr_P(lqr, K):
    """
    Computes the P matrix for a given policy K.
    The value function is the result of V(s) = -s.T @ P @ s

    Args:
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix

    Returns:
        The P matrix of the value function

    """
    A, B, Q, R, gamma = _parse_lqr(lqr)

    L, M = _compute_lqr_intermediate_results(K, A, B, Q, R, gamma)

    vec_P = np.linalg.solve(M, L.reshape(-1))

    return vec_P.reshape(Q.shape)


def compute_lqr_V(s, lqr, K):
    """
    Computes the value function at a state x, with the given controller matrix K.
    Convert s if shape is not (n_samples, n_features)

    Args:
        s (np.ndarray): state
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix

    Returns:
        The value function at s

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))

    P = compute_lqr_P(lqr, K)
    m = lambda x, y: x.T @ P @ x
    return -1. * pairwise_distances(s, metric=m).diagonal().reshape((-1, 1))


def compute_lqg_V(s, lqr, K, Sigma):
    """
    Computes the value function at a state x, with the given controller matrix K and covariance Sigma.
    Convert s if shape is not (n_samples, n_features)
    Convert s if shape is not (n_samples, n_features)

    Args:
        s (np.ndarray): state
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix
        Sigma (np.ndarray): covariance matrix

    Returns:
        The value function at s

    """
    P = compute_lqr_P(lqr, K)
    A, B, Q, R, gamma = _parse_lqr(lqr)

    return compute_lqr_V(s, lqr, K) - np.trace(Sigma @ (R + gamma * B.T @ P @ B)) / (1.0 - gamma)


def compute_lqr_Q(s, a, lqr, K):
    """
    Computes the state-action value function Q at a state-action pair x,
    with the given controller matrix K.
    Convert s if shape is not (n_samples, n_features)

    Args:
        s (np.ndarray): state
        a (np.ndarray): action
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix

    Returns:
        The Q function at s, a

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))
    if a.ndim == 1:
        a = a.reshape((1, -1))
    sa = np.hstack((s, a))

    M = _compute_lqr_Q_matrix(lqr, K)
    m = lambda x, y: x.T @ M @ x
    return -1. * pairwise_distances(sa, metric=m).diagonal().reshape((-1, 1))


def compute_lqg_Q(s, a, lqr, K, Sigma):
    """
    Computes the state-action value function Q at a state-action pair x,
    with the given controller matrix K and covariance Sigma.
    Convert s if shape is not (n_samples, n_features)

    Args:
        s (np.ndarray): state-action pair
        a (np.ndarray): action
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix
        Sigma (np.ndarray): covariance matrix

    Returns:
        The Q function at s, a

    """
    b = _compute_lqg_Q_additional_term(lqr, K, Sigma)
    return compute_lqr_Q(s, a, lqr, K) - b


def compute_lqg_gradient(s, lqr, K, Sigma):
    """
    Computes the gradient of the objective function J at state s, w.r.t. the controller matrix K, with the current
    policy parameters K and Sigma.
    J(s, K, Sigma) = ValueFunction(s, K, Sigma)
    Convert s if shape is not (n_samples, n_features)

    Args:
        s (np.ndarray): state pair
        lqr (LQR): LQR environment
        K (np.ndarray): controller matrix
        Sigma (np.ndarray): covariance matrix

    Returns:
        The gradient of J w.r.t. to K

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))
    batch_size = s.shape[0]

    A, B, Q, R, gamma = _parse_lqr(lqr)
    L, M = _compute_lqr_intermediate_results(K, A, B, Q, R, gamma)

    Minv = np.linalg.inv(M)

    n_elems = K.shape[0]*K.shape[1]
    dJ = np.zeros((batch_size, n_elems))
    for i in range(n_elems):
        dLi, dMi = _compute_lqr_intermediate_results_diff(K, A, B, R, gamma, i)

        vec_dPi = -Minv @ dMi @ Minv @ L.reshape(-1) + np.linalg.solve(M, dLi.reshape(-1))

        dPi = vec_dPi.reshape(Q.shape)

        m = lambda x, y: x.T @ dPi @ x

        dJ[:, i] = pairwise_distances(s, metric=m).diagonal().reshape((-1, 1)) \
                    + gamma * np.trace(Sigma @ B.T @ dPi @ B) / (1.0 - gamma)

    return -dJ


def _parse_lqr(lqr):
    return lqr.A, lqr.B, lqr.Q, lqr.R, lqr.info.gamma


def _compute_riccati_rhs(A, B, Q, R, gamma, K, P):
    return Q + gamma*(A.T @ P @ A - K.T @ B.T @ P @ A - A.T @ P @ B @ K + K.T @ B.T @ P @ B @ K) \
           + K.T @ R @ K


def _compute_riccati_gain(P, A, B, R, gamma):
    return gamma * np.linalg.inv((R + gamma * (B.T @ P @ B))) @ B.T @ P @ A


def _compute_lqr_intermediate_results(K, A, B, Q, R, gamma):
    size = Q.shape[0] ** 2

    L = Q + K.T @ R @ K
    kb = K.T @ B.T
    M = np.eye(size, size) - gamma * (np.kron(A.T, A.T) - np.kron(A.T, kb) - np.kron(kb, A.T) + np.kron(kb, kb))

    return L, M


def _compute_lqr_intermediate_results_diff(K, A, B, R, gamma, i):
    n_elems = K.shape[0]*K.shape[1]
    vec_dKi = np.zeros(n_elems)
    vec_dKi[i] = 1
    dKi = vec_dKi.reshape(K.shape)
    kb = K.T @ B.T
    dkb = dKi.T @ B.T

    dL = dKi.T @ R @ K + K.T @ R @ dKi
    dM = gamma * (np.kron(A.T, dkb) + np.kron(dkb, A.T) - np.kron(dkb, kb) - np.kron(kb, dkb))

    return dL, dM


def _compute_lqr_Q_matrix(lqr, K):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    P = compute_lqr_P(lqr, K)

    M = np.block([[Q + gamma * A.T @ P @ A, gamma * A.T @ P @ B],
                  [gamma * B.T @ P @ A, R + gamma * B.T @ P @ B]])

    return M


def _compute_lqg_Q_additional_term(lqr, K, Sigma):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    P = compute_lqr_P(lqr, K)
    b = gamma/(1-gamma)*np.trace(Sigma @ (R + gamma * B.T @ P @ B))
    return b
