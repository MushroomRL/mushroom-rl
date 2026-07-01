import numpy as np
from copy import deepcopy

from mushroom_rl.policy.policy import Policy, HasWeights


class VectorPolicy(Policy, HasWeights):
    """
    Policy wrapping a vector of independent copies of a base policy, each one with its own weights. It is used by
    black-box optimization algorithms to evaluate a population of parameterizations in parallel, one per environment.
    Each wrapped policy manages its own internal state (if stateful), so no policy state is threaded through this
    wrapper.

    """
    def __init__(self, policy, n_envs):
        """
        Constructor.

        Args:
            policy (HasWeights): base policy to copy;
            n_envs (int): number of environments to be repeated.

        """
        self._policy_vector = [deepcopy(policy) for _ in range(n_envs)]

        self._add_save_attr(_policy_vector='mushroom')

    def draw_action(self, state):
        actions = list()
        for i, policy in enumerate(self._policy_vector):
            actions.append(policy.draw_action(state[i]))

        return np.array(actions)

    def set_n(self, n_envs):
        if len(self) > n_envs:
            self._policy_vector = self._policy_vector[:n_envs]
        elif len(self) < n_envs:
            n_missing = n_envs - len(self)
            self._policy_vector += [deepcopy(self._policy_vector[0]) for _ in range(n_missing)]

    def get_flat_policy(self):
        return self._policy_vector[0]

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        for i, policy in enumerate(self._policy_vector):
            policy.set_weights(weights[i])

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """

        weight_list = list()
        for i, policy in enumerate(self._policy_vector):
            weights_i = policy.get_weights()
            weight_list.append(weights_i)

        return np.array(weight_list)

    @property
    def weights_size(self):
        """
        Property.

        Returns:
             The size of the policy weights.

        """
        return len(self), self._policy_vector[0].weights_size

    def reset(self):
        for policy in self._policy_vector:
            policy.reset()

    def reset_vectorized(self, start_mask):
        for masked, policy in zip(start_mask, self._policy_vector):
            if masked:
                policy.reset()

    def __len__(self):
        return len(self._policy_vector)
