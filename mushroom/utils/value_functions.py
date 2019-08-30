import numpy as np


def compute_gae(V, s, ss, r, absorbing, last, gamma, lam):
    """
    Function to compute Generalized Advantage Estimation (GAE)
    and new value function target over a dataset.

    "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation".
    Schulman J. et al.. 2016.

    Args:
        V (Regressor): the current value function regressor;
        s (numpy.ndarray): the set of states in which we want
            to evaluate the advantage;
        ss (numpy.ndarray): the set of next states in which we want
            to evaluate the advantage;
        r (numpy.ndarray): the reward obtained in each transition
            from state s to state ss;
        absorbing (numpy.ndarray): an array of boolean flags indicating
            if the reached state is absorbing;
        absorbing (numpy.ndarray): an array of boolean flags indicating
            if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA
            algorithm.
    """
    v = V(s)
    v_next = V(ss)
    gen_adv = np.empty_like(v)
    for rev_k, _ in enumerate(reversed(v)):
        k = len(v) - rev_k - 1
        if last[k] or rev_k == 0:
            gen_adv[k] = r[k] - v[k]
            if not absorbing[k]:
                gen_adv[k] += gamma * v_next[k]
        else:
            gen_adv[k] = r[k] + gamma * v_next[k] - v[k] + gamma * lam * gen_adv[k + 1]
    return gen_adv + v, gen_adv