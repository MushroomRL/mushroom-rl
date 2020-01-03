import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP


def generate_simple_chain(state_n, goal_states, prob, rew, mu=None, gamma=.9,
                          horizon=100):
    """
    Simple chain generator.

    Args:
        state_n (int): number of states;
        goal_states (list): list of goal states;
        prob (float): probability of success of an action;
        rew (float): reward obtained in goal states;
        mu (np.ndarray): initial state probability distribution;
        gamma (float, .9): discount factor;
        horizon (int, 100): the horizon.

    Returns:
        A FiniteMDP object built with the provided parameters.

    """
    p = compute_probabilities(state_n, prob)
    r = compute_reward(state_n, goal_states, rew)

    assert mu is None or len(mu) == state_n

    return FiniteMDP(p, r, mu, gamma, horizon)


def compute_probabilities(state_n, prob):
    """
    Compute the transition probability matrix.

    Args:
        state_n (int): number of states;
        prob (float): probability of success of an action.

    Returns:
        The transition probability matrix;

    """
    p = np.zeros((state_n, 2, state_n))

    for i in range(state_n):
        if i == 0:
            p[i, 1, i] = 1.
        else:
            p[i, 1, i] = 1. - prob
            p[i, 1, i - 1] = prob

        if i == state_n - 1:
            p[i, 0, i] = 1.
        else:
            p[i, 0, i] = 1. - prob
            p[i, 0, i + 1] = prob

    return p


def compute_reward(state_n, goal_states, rew):
    """
    Compute the reward matrix.

    Args:
        state_n (int): number of states;
        goal_states (list): list of goal states;
        rew (float): reward obtained in goal states.

    Returns:
        The reward matrix.

    """
    r = np.zeros((state_n, 2, state_n))

    for g in goal_states:
        if g != 0:
            r[g - 1, 0, g] = rew

        if g != state_n - 1:
            r[g + 1, 1, g] = rew

    return r
