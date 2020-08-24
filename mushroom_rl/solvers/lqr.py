import numpy as np


def compute_lqr_feedback_gain(lqr, max_iterations=100):
    """
    Computes the optimal gain matrix K.

    Args:
        lqr (LQR): LQR environment;
        max_iterations (int): max iterations for convergence.

    Returns:
        Feedback gain matrix K.

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
    Computes the P matrix for a given gain matrix K.

    Args:
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix.

    Returns:
        The P matrix of the value function.

    """
    A, B, Q, R, gamma = _parse_lqr(lqr)

    L, M = _compute_lqr_intermediate_results(K, A, B, Q, R, gamma)

    vec_P = np.linalg.solve(M, L.reshape(-1))

    return vec_P.reshape(Q.shape)


def compute_lqr_V(s, lqr, K):
    """
    Computes the value function at a state s, with the given gain matrix K.

    Args:
        s (np.ndarray): state;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix.

    Returns:
        The value function at s

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))

    P = compute_lqr_P(lqr, K)
    return -1. * np.einsum('...k,kl,...l->...', s, P, s).reshape(-1, 1)


def compute_lqr_V_gaussian_policy(s, lqr, K, Sigma):
    """
    Computes the value function at a state s, with the given gain matrix K and
    covariance Sigma.

    Args:
        s (np.ndarray): state;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix;
        Sigma (np.ndarray): covariance matrix.

    Returns:
        The value function at s.

    """
    b = _compute_lqr_V_gaussian_policy_additional_term(lqr, K, Sigma)
    return compute_lqr_V(s, lqr, K) - b


def compute_lqr_Q(s, a, lqr, K):
    """
    Computes the state-action value function Q at a state-action pair (s, a),
    with the given gain matrix K.

    Args:
        s (np.ndarray): state;
        a (np.ndarray): action;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix.

    Returns:
        The Q function at s, a.

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))
    if a.ndim == 1:
        a = a.reshape((1, -1))
    sa = np.hstack((s, a))

    M = _compute_lqr_Q_matrix(lqr, K)

    return -1. * np.einsum('...k,kl,...l->...', sa, M, sa).reshape(-1, 1)


def compute_lqr_Q_gaussian_policy(s, a, lqr, K, Sigma):
    """
    Computes the state-action value function Q at a state-action pair (s, a),
    with the given gain matrix K and covariance Sigma.

    Args:
        s (np.ndarray): state;
        a (np.ndarray): action;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix;
        Sigma (np.ndarray): covariance matrix.

    Returns:
        The Q function at (s, a).

    """
    b = _compute_lqr_Q_gaussian_policy_additional_term(lqr, K, Sigma)
    return compute_lqr_Q(s, a, lqr, K) - b


def compute_lqr_V_gaussian_policy_gradient_K(s, lqr, K, Sigma):
    """
    Computes the gradient of the objective function J (equal to the value
    function V) at state s, w.r.t. the controller matrix K, with the current
    policy parameters K and Sigma. J(s, K, Sigma) = ValueFunction(s, K, Sigma).

    Args:
        s (np.ndarray): state;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix;
        Sigma (np.ndarray): covariance matrix.

    Returns:
        The gradient of J wrt to K.

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

        vec_dPi = -Minv @ dMi @ Minv @ L.reshape(-1) + np.linalg.solve(
            M, dLi.reshape(-1)
        )

        dPi = vec_dPi.reshape(Q.shape)

        dJ[:, i] = np.einsum('...k,kl,...l->...', s, dPi, s).reshape(-1, 1) \
                   + gamma * np.trace(Sigma @ B.T @ dPi @ B) / (1.0 - gamma)

    return -dJ


def compute_lqr_Q_gaussian_policy_gradient_K(s, a, lqr, K, Sigma):
    """
    Computes the gradient of the state-action Value function at state-action
    pair (s, a), w.r.t. the controller matrix K, with the current policy
    parameters K and Sigma.

    Args:
        s (np.ndarray): state;
        a (np.ndarray): action;
        lqr (LQR): LQR environment;
        K (np.ndarray): controller matrix;
        Sigma (np.ndarray): covariance matrix.

    Returns:
        The gradient of Q wrt to K.

    """
    if s.ndim == 1:
        s = s.reshape((1, -1))
    if a.ndim == 1:
        a = a.reshape((1, -1))

    s_next = (lqr.A @ s.T).T + (lqr.B @ a.T).T

    return lqr.info.gamma * compute_lqr_V_gaussian_policy_gradient_K(
        s_next, lqr, K, Sigma
    )


def _parse_lqr(lqr):
    return lqr.A, lqr.B, lqr.Q, lqr.R, lqr.info.gamma


def _compute_riccati_rhs(A, B, Q, R, gamma, K, P):
    return Q + gamma * (
            A.T @ P @ A - K.T @ B.T @ P @ A - A.T @ P @ B @ K +
            K.T @ B.T @ P @ B @ K) + K.T @ R @ K


def _compute_riccati_gain(P, A, B, R, gamma):
    return gamma * np.linalg.inv((R + gamma * (B.T @ P @ B))) @ B.T @ P @ A


def _compute_lqr_intermediate_results(K, A, B, Q, R, gamma):
    size = Q.shape[0] ** 2

    L = Q + K.T @ R @ K
    kb = K.T @ B.T
    M = np.eye(size, size) - gamma * (np.kron(A.T, A.T) - np.kron(A.T, kb) -
                                      np.kron(kb, A.T) + np.kron(kb, kb))

    return L, M


def _compute_lqr_intermediate_results_diff(K, A, B, R, gamma, i):
    n_elems = K.shape[0]*K.shape[1]
    vec_dKi = np.zeros(n_elems)
    vec_dKi[i] = 1
    dKi = vec_dKi.reshape(K.shape)
    kb = K.T @ B.T
    dkb = dKi.T @ B.T

    dL = dKi.T @ R @ K + K.T @ R @ dKi
    dM = gamma * (np.kron(A.T, dkb) + np.kron(dkb, A.T) - np.kron(dkb, kb) -
                  np.kron(kb, dkb))

    return dL, dM


def _compute_lqr_Q_matrix(lqr, K):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    P = compute_lqr_P(lqr, K)

    M = np.block([[Q + gamma * A.T @ P @ A, gamma * A.T @ P @ B],
                  [gamma * B.T @ P @ A, R + gamma * B.T @ P @ B]])

    return M


def _compute_lqr_V_gaussian_policy_additional_term(lqr, K, Sigma):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    P = compute_lqr_P(lqr, K)
    b = np.trace(Sigma @ (R + gamma * B.T @ P @ B)) / (1.0 - gamma)

    return b


def _compute_lqr_Q_gaussian_policy_additional_term(lqr, K, Sigma):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    P = compute_lqr_P(lqr, K)
    b = gamma / (1 - gamma) * np.trace(Sigma @ (R + gamma * B.T @ P @ B))

    return b
