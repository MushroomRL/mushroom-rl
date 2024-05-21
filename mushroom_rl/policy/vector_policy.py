import numpy as np
from copy import deepcopy

from .policy import ParametricPolicy


class VectorPolicy(ParametricPolicy):
    def __init__(self, policy, n_envs):
        """
        Constructor.

        Args:
            policy (ParametricPolicy): base policy to copy
            n_envs: number of environments to be repeated.

        """
        super().__init__(policy_state_shape=policy.policy_state_shape)
        self._policy_vector = [deepcopy(policy) for _ in range(n_envs)]

        self._add_save_attr(_policy_vector='mushroom')

    def draw_action(self, state, policy_state):
        actions = list()
        policy_next_states = list()
        for i, policy in enumerate(self._policy_vector):
            s = state[i]
            ps = policy_state[i] if policy_state is not None else None
            action, policy_next_state = policy.draw_action(s, policy_state=ps)

            actions.append(action)

            if policy_next_state is not None:
                policy_next_state.append(policy_next_state)

        return np.array(actions), None if len(policy_next_states) == 0 else np.array(policy_next_state)

    def set_n(self, n_envs):
        if len(self) < n_envs:
            self._policy_vector = self._policy_vector[:n_envs]
        if len(self) > n_envs:
            n_missing = n_envs - len(self)
            self._policy_vector += [self._policy_vector[0] for _ in range(n_missing)]

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

    def reset(self, mask=None):
        policy_states = None
        if self.policy_state_shape is None:
            if mask is None:
                for policy in self._policy_vector:
                    policy.reset()
            else:
                for masked, policy in zip(mask, self._policy_vector):
                    if masked:
                        policy.reset()
        else:
            policy_states = np.empty((len(self._policy_vector),) + self.policy_state_shape)
            if mask is None:
                for i, policy in enumerate(self._policy_vector):
                    policy_states[i] = policy.reset()
            else:
                for i, (masked, policy) in enumerate(zip(mask, self._policy_vector)):
                    if masked:
                        policy_states[i] = policy.reset()
        return policy_states

    def __len__(self):
        return len(self._policy_vector)


