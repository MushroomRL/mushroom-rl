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
    n_states = p.shape[0]
    n_actions = p.shape[1]

    v = np.zeros(n_states)

    while True:
        v_old = deepcopy(v)

        for s in xrange(n_states):
            vmax = -np.inf
            for a in xrange(n_actions):
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
    n_states = p.shape[0]
    n_actions = p.shape[1]

    pi = np.zeros(n_states, dtype=int)
    v = np.zeros(n_states)

    changed = True
    while changed:
        p_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)
        i = np.eye(n_states)

        for s in xrange(n_states):
            a = pi[s]
            p_pi_s = p[s, a, :]
            r_pi_s = r[s, a, :]

            p_pi[s, :] = p_pi_s.T
            r_pi[s] = p_pi_s.T.dot(r_pi_s)

        v = np.linalg.inv(i - gamma * p_pi).dot(r_pi)

        changed = False

        for s in xrange(n_states):
            vmax = v[s]
            for a in xrange(n_actions):
                if a != pi[s]:
                    p_sa = p[s, a]
                    r_sa = r[s, a]
                    va = p_sa.T.dot(r_sa + gamma * v)
                    if va > vmax:
                        pi[s] = a
                        vmax = va
                        changed = True

    return v, pi
