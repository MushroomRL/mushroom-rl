import numpy as np


def numerical_diff_policy(policy, state, action, eps=1e-6):

    w_start = policy.get_weights()

    g = np.zeros(policy.weights_size)
    for i in range(len(w_start)):
        perturb = np.zeros(policy.weights_size)
        perturb[i] = eps

        policy.set_weights(w_start - perturb)
        v1 = policy(state, action)

        policy.set_weights(w_start + perturb)
        v2 = policy(state, action)

        g[i] = (v2 - v1) / (2 * eps)

    policy.set_weights(w_start)

    return g


def numerical_diff_dist(dist, theta, eps=1e-6):

    rho_start = dist.get_parameters()

    g = np.zeros(dist.parameters_size)
    for i in range(len(rho_start)):
        perturb = np.zeros(dist.parameters_size)
        perturb[i] = eps

        dist.set_parameters(rho_start - perturb)
        v1 = dist(theta)

        dist.set_parameters(rho_start + perturb)
        v2 = dist(theta)

        g[i] = (v2 - v1) / (2 * eps)

    dist.set_parameters(rho_start)

    return g