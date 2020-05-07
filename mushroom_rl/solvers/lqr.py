import numpy as np


def _parse_lqr(lqr):
    return lqr.A, lqr.B, lqr.Q, lqr.R, lqr.info.gamma


def _compute_riccati_rhs(A, B, Q, R, gamma, K, P):

    return Q + gamma*(A.T @ P @ A - K @ B.T @ P @ A - A.T @ P @ B @ K.T + K @ B.T @ P @ B @ K.T) \
           + K @ R @ K.T


def _compute_riccati_gain(P, A, B, R, gamma):
    return gamma * np.linalg.inv((R + gamma * (B.T @ P @ B))) @ B.T @ P @ A


def solve_lqr_linear(lqr, max_iterations=100):
    A, B, Q, R, gamma = _parse_lqr(lqr)

    P = np.eye(Q.shape[0])
    K = _compute_riccati_gain(P, A, B, R, gamma)

    it = 0
    while it < max_iterations:
        P = _compute_riccati_rhs(A, B, Q, R, gamma, K, P)
        K = _compute_riccati_gain(P, A, B, R, gamma)

        it += 1

    return K
