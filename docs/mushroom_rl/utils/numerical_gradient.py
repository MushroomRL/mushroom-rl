import numpy as np


def numerical_diff_policy(policy, state, action, eps=1e-6):
    """
    Compute the gradient of a policy in (``state``, ``action``) numerically.

    Args:
        policy (Policy): the policy whose gradient has to be returned;
        state (np.ndarray): the state;
        action (np.ndarray): the action;
        eps (float, 1e-6): the value of the perturbation.

    Returns:
        The gradient of the provided policy in (``state``, ``action``)
        computed numerically.

    """
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
    """
    Compute the gradient of a distribution in ``theta`` numerically.

    Args:
        dist (Distribution): the distribution whose gradient has to be returned;
        theta (np.ndarray): the parametrization where to compute the gradient;
        eps (float, 1e-6): the value of the perturbation.

    Returns:
        The gradient of the provided distribution ``theta`` computed
        numerically.

    """
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


def numerical_diff_function(function, params, eps=1e-6):
    """
    Compute the gradient of a function in ``theta`` numerically.

    Args:
        function: a function whose gradient has to be returned;
        params: parameter vector w.r.t. we need to compute the gradient;
        eps (float, 1e-6): the value of the perturbation.

    Returns:
        The numerical  gradient of the function computed w.r.t. parameters
        ``params``.

    """

    g = np.zeros_like(params)
    n_params = len(params)

    for i in range(n_params):
        perturb = np.zeros(n_params)
        perturb[i] = eps

        v1 = function(params - perturb)
        v2 = function(params + perturb)

        g[i] = (v2 - v1) / (2 * eps)

    return g