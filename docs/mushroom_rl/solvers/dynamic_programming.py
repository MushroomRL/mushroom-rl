import numpy as np
from copy import deepcopy


def value_iteration(prob, reward, gamma, eps):
    """
    Value iteration algorithm to solve a dynamic programming problem.

    Args:
        prob (np.ndarray): transition probability matrix;
        reward (np.ndarray): reward matrix;
        gamma (float): discount factor;
        eps (float): accuracy threshold.

    Returns:
        The optimal value of each state.

    """
    n_states = prob.shape[0]
    n_actions = prob.shape[1]

    value = np.zeros(n_states)

    while True:
        value_old = deepcopy(value)

        for state in range(n_states):
            vmax = -np.inf
            for action in range(n_actions):
                prob_state_action = prob[state, action, :]
                reward_state_action = reward[state, action, :]
                va = prob_state_action.T.dot(
                    reward_state_action + gamma * value_old)
                vmax = max(va, vmax)

            value[state] = vmax
        if np.linalg.norm(value - value_old) <= eps:
            break

    return value


def policy_iteration(prob, reward, gamma):
    """
    Policy iteration algorithm to solve a dynamic programming problem.

    Args:
        prob (np.ndarray): transition probability matrix;
        reward (np.ndarray): reward matrix;
        gamma (float): discount factor.

    Returns:
        The optimal value of each state and the optimal policy.

    """
    n_states = prob.shape[0]
    n_actions = prob.shape[1]

    policy = np.zeros(n_states, dtype=int)
    value = np.zeros(n_states)

    changed = True
    while changed:
        p_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)
        i = np.eye(n_states)

        for state in range(n_states):
            action = policy[state]
            p_pi_s = prob[state, action, :]
            r_pi_s = reward[state, action, :]

            p_pi[state, :] = p_pi_s.T
            r_pi[state] = p_pi_s.T.dot(r_pi_s)

        value = np.linalg.solve(i - gamma * p_pi, r_pi)

        changed = False

        for state in range(n_states):
            vmax = value[state]
            for action in range(n_actions):
                if action != policy[state]:
                    p_sa = prob[state, action]
                    r_sa = reward[state, action]
                    va = p_sa.T.dot(r_sa + gamma * value)
                    if va > vmax and not np.isclose(va, vmax):
                        policy[state] = action
                        vmax = va
                        changed = True

    return value, policy
