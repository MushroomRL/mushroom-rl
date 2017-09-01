import numpy as np

from mushroom.environments.finite_mdp import FiniteMDP


def generate_simple_chain(state_n, goal_states, prob, rew, mu=None, gamma=.9):
    p = compute_probabilities(state_n, prob)
    r = compute_reward(state_n, goal_states, rew)

    return FiniteMDP(p, r, mu, gamma)


def compute_probabilities(state_n, prob):
    p = np.zeros((state_n, 2, state_n))

    for i in xrange(state_n):
        if i == 0:
            p[i, 1, i] = 1.0
        else:
            p[i, 1, i] = 1.0 - prob
            p[i, 1, i - 1] = prob

        if i == state_n - 1:
            p[i, 0, i] = 1.0
        else:
            p[i, 0, i] = 1.0 - prob
            p[i, 0, i + 1] = prob

    return p


def compute_reward(state_n, goal_states, rew):
    r = np.zeros((state_n, 2, state_n))

    for g in goal_states:
        if g != 0:
            r[g - 1, 0, g] = rew

        if g != state_n - 1:
            r[g + 1, 1, g] = rew

    return r
