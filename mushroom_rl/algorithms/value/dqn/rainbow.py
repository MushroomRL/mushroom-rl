from copy import deepcopy

import torch

from mushroom_rl.algorithms.value.dqn.categorical_dqn import AbstractCategoricalDQN
from mushroom_rl.approximators.parametric.networks import RainbowNetwork
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory


class Rainbow(AbstractCategoricalDQN):
    """
    Rainbow algorithm.
    "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    Hessel M. et al. 2018.

    """
    def __init__(self, mdp_info, policy, approximator_params, n_atoms, v_min,
                 v_max, n_steps_return, alpha_coeff, beta, sigma_coeff=.5,
                 **params):
        """
        Constructor.

        Args:
            n_atoms (int): number of atoms;
            v_min (float): minimum value of value-function;
            v_max (float): maximum value of value-function;
            n_steps_return (int): the number of steps to consider to compute the n-return;
            alpha_coeff (float): prioritization exponent for prioritized experience replay;
            beta (Parameter): importance sampling coefficient for prioritized experience replay;
            sigma_coeff (float, .5): sigma0 coefficient for noise initialization in noisy layers.

        """
        features_network = approximator_params['network']
        approximator_params = deepcopy(approximator_params)
        approximator_params['network'] = RainbowNetwork
        approximator_params['features_network'] = features_network
        approximator_params['n_atoms'] = n_atoms
        approximator_params['v_min'] = v_min
        approximator_params['v_max'] = v_max
        approximator_params['sigma_coeff'] = sigma_coeff

        self._n_steps_return = n_steps_return
        self._sigma_coeff = sigma_coeff

        params.setdefault('replay_memory', {"class": PrioritizedReplayMemory,
                                            "params": dict(alpha=alpha_coeff, beta=beta,
                                                           n_steps_return=n_steps_return)})

        super().__init__(mdp_info, policy, approximator_params, n_atoms, v_min, v_max, **params)

        self._add_save_attr(
            _n_steps_return='primitive',
            _sigma_coeff='primitive'
        )

    def _compute_target(self, reward, next_state, absorbing):
        q_next = self.approximator.predict(next_state, **self._predict_params)
        a_max = torch.argmax(q_next, 1).unsqueeze(1)
        gamma = self.mdp_info.gamma ** self._n_steps_return * ~absorbing
        p_next = self.target_approximator.predict(next_state, a_max, get_distribution=True, **self._predict_params)

        return self._categorical_projection(reward, gamma, p_next)
