import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.policy import VectorPolicy

from mushroom_rl.algorithms.policy_search.black_box_optimization.context_builder import ContextBuilder


class BlackBoxOptimization(Agent):
    """
    Base class for black box optimization algorithms.
    These algorithms work on a distribution of policy parameters and often they
    do not rely on stochastic and differentiable policies.

    """
    def __init__(self, mdp_info, distribution, policy, context_builder=None, backend='numpy'):
        """
        Constructor.

        Args:
            distribution (Distribution): the distribution of policy parameters;
            policy (ParametricPolicy): the policy to use;
            context_builder (ContextBuilder, None): class used to compute the context variables from initial state and
                the episode_info dictionary;
            backend (str, 'numpy'): the backend used by the algorithm.

        """
        assert (not distribution.is_contextual and context_builder is None) or \
               (distribution.is_contextual and context_builder is not None)
        self.distribution = distribution
        self._context_builder = ContextBuilder() if context_builder is None else context_builder
        self._deterministic = False

        super().__init__(mdp_info, policy, is_episodic=True, backend=backend)

        self._add_save_attr(
            distribution='mushroom',
            _context_builder='mushroom',
            _deterministic='primitive'
        )

    def episode_start(self, initial_state, episode_info):
        if isinstance(self.policy, VectorPolicy):
            self.policy = self.policy.get_flat_policy()

        context = self._context_builder(initial_state, **episode_info)

        if self._deterministic:
            theta = self.distribution.mean(context)
        else:
            theta = self.distribution.sample(context)
        self.policy.set_weights(theta)

        policy_state, _ = super().episode_start(initial_state, episode_info)

        return policy_state, theta

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        n_envs = len(start_mask)
        if not isinstance(self.policy, VectorPolicy):
            self.policy = VectorPolicy(self.policy, n_envs)
        elif len(self.policy) != n_envs:
            self.policy.set_n(n_envs)

        theta = self.policy.get_weights()
        if start_mask.any():
            context = self._context_builder(initial_states, **episode_info)[start_mask]

            if self._deterministic:
                if context is not None:
                    theta[start_mask] = self.distribution.mean(context[start_mask])
                else:
                    theta[start_mask] = self._agent_backend.from_list(
                        [self.distribution.mean() for _ in range(start_mask.sum())])
            else:
                if context is not None:
                    theta[start_mask] = self._agent_backend.from_list(
                        [self.distribution.sample(context[i]) for i in range(start_mask.sum())])  # TODO change it
                else:
                    theta[start_mask] = self._agent_backend.from_list(
                        [self.distribution.sample() for _ in range(start_mask.sum())])
            self.policy.set_weights(theta)

        policy_states = self.policy.reset()

        return policy_states, theta

    def fit(self, dataset):
        Jep = dataset.discounted_return
        theta = self._agent_backend.from_list(dataset.theta_list)

        if self.distribution.is_contextual:
            initial_states = dataset.get_init_states()
            episode_info = dataset.episode_info

            context = self._context_builder(initial_states, **episode_info)
        else:
            context = None

        self._update(Jep, theta, context)

    def set_deterministic(self, deterministic=True):
        self._deterministic = deterministic

    def _update(self, Jep, theta, context):
        """
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the
        proper update.

        Args:
            Jep (np.ndarray): a vector containing the J of the considered
                trajectories;
            theta (np.ndarray): a matrix of policy parameters of the considered
                trajectories.

        """
        raise NotImplementedError('BlackBoxOptimization is an abstract class')
