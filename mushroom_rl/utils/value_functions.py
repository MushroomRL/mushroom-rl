import numpy as np


def compute_advantage_montecarlo(V, s, ss, r, absorbing, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using rollouts
    (monte carlo estimation).

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
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    r = r.squeeze()
    q = np.zeros(len(r))
    v = V(s).squeeze()

    q_next = V(ss[-1]).squeeze().item()
    for rev_k in range(len(r)):
        k = len(r) - rev_k - 1
        q_next = r[k] + gamma * q_next * (1. - absorbing[k])
        q[k] = q_next

    adv = q - v
    return q[:, np.newaxis], adv[:, np.newaxis]


def compute_advantage(V, s, ss, r, absorbing, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using bootstrapping.

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
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    v = V(s).squeeze()
    v_next = V(ss).squeeze() * (1 - absorbing)

    q = r + gamma * v_next
    adv = q - v
    return q[:, np.newaxis], adv[:, np.newaxis]


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
        last (numpy.ndarray): an array of boolean flags indicating
            if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA
            algorithm.
    Returns:
        The new estimate for the value function of the next state
        and the estimated generalized advantage.
    """
    v = V(s)
    v_next = V(ss)
    gen_adv = np.empty_like(v)
    for rev_k in range(len(v)):
        k = len(v) - rev_k - 1
        if last[k] or rev_k == 0:
            gen_adv[k] = r[k] - v[k]
            if not absorbing[k]:
                gen_adv[k] += gamma * v_next[k]
        else:
            gen_adv[k] = r[k] + gamma * v_next[k] - v[k] + gamma * lam * gen_adv[k + 1]
    return gen_adv + v, gen_adv