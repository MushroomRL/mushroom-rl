import numpy as np
from copy import deepcopy


def value_iteration(p, r, gamma, eps):
    """
    Value iteration algorithm to solve a dynamic programming problem.

    Args:
        p (np.array): transition probability matrix;
        r (np.array): reward matrix;
        gamma (float): discount factor;
        eps (float): accuracy threshold.

    """
    state_n = p.shape[0]
    action_n = p.shape[1]

    v = np.zeros(state_n)

    while True:
        v_old = deepcopy(v)

        for s in xrange(state_n):
            vmax = -float('inf')
            for a in xrange(action_n):
                p_sa = p[s, a, :]
                r_sa = r[s, a, :]
                va = p_sa.T.dot(r_sa + gamma * v_old)
                vmax = max(va, vmax)

            v[s] = vmax
        if np.linalg.norm(v - v_old) <= eps:
            break

    return v


def policy_iteration(p, r, gamma):
    """
    Policy iteration algorithm to solve a dynamic programming problem.

    Args:
        p (np.array): transition probability matrix;
        r (np.array): reward matrix;
        gamma (float): discount factor;

    """
    state_n = p.shape[0]
    action_n = p.shape[1]

    pi = np.zeros(state_n, dtype=int)
    v = np.zeros(state_n)

    changed = True
    while changed:
        p_pi = np.zeros((state_n, state_n))
        r_pi = np.zeros(state_n)
        i = np.eye(state_n)

        for s in xrange(state_n):
            a = pi[s]
            p_pi_s = p[s, a, :]
            r_pi_s = r[s, a, :]

            p_pi[s, :] = p_pi_s.T
            r_pi[s] = p_pi_s.T.dot(r_pi_s)

        v = np.linalg.inv(i - gamma * p_pi).dot(r_pi)

        changed = False

        for s in xrange(state_n):
            vmax = v[s]
            for a in xrange(action_n):
                if a != pi[s]:
                    p_sa = p[s, a]
                    r_sa = r[s, a]
                    va = p_sa.T.dot(r_sa + gamma * v)
                    if va > vmax:
                        pi[s] = a
                        vmax = va
                        changed = True

    return v, pi
