from copy import deepcopy

import torch

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.algorithms.value.dqn.categorical_dqn import categorical_loss
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import RainbowNetwork
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.utils.torch import TorchUtils

eps = torch.finfo(torch.float32).eps


class Rainbow(AbstractDQN):
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
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = RainbowNetwork
        params['approximator_params']['features_network'] = features_network
        params['approximator_params']['n_atoms'] = n_atoms
        params['approximator_params']['v_min'] = v_min
        params['approximator_params']['v_max'] = v_max
        params['approximator_params']['sigma_coeff'] = sigma_coeff
        params['approximator_params']['loss'] = categorical_loss

        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        self._delta = (v_max - v_min) / (n_atoms - 1)
        self._a_values = torch.arange(v_min, v_max + eps, self._delta, device=TorchUtils.get_device())
        self._n_steps_return = n_steps_return
        self._sigma_coeff = sigma_coeff
        self._pending = None

        params['replay_memory'] = {"class": PrioritizedReplayMemory,
                                   "params": dict(alpha=alpha_coeff, beta=beta,
                                                  n_steps_return=n_steps_return)}

        super().__init__(mdp_info, policy, TorchApproximator, **params)

        self._add_save_attr(
            _n_atoms='primitive',
            _v_min='primitive',
            _v_max='primitive',
            _delta='primitive',
            _a_values='torch',
            _n_steps_return='primitive',
            _sigma_coeff='primitive',
            _pending='none'
        )

    def fit(self, dataset):
        if self._pending is not None:
            dataset = self._pending + dataset
        self._pending = dataset[-(self._n_steps_return - 1):] if self._n_steps_return > 1 else None
        initial_priority = torch.ones(len(dataset), device=TorchUtils.get_device()) * self._replay_memory.max_priority
        self._replay_memory.add(dataset, initial_priority)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_, idxs, is_weight = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = torch.clip(reward, -1, 1)

            with torch.no_grad():
                q_next = self.approximator.predict(next_state, **self._predict_params)
                a_max = torch.argmax(q_next, 1)
                gamma = self.mdp_info.gamma ** self._n_steps_return * ~absorbing
                p_next = self.target_approximator.predict(next_state, a_max,
                                                          get_distribution=True, **self._predict_params)
                gamma_z = gamma.unsqueeze(1) * self._a_values
                bell_a = (reward.unsqueeze(1) + gamma_z).clip(self._v_min,
                                                              self._v_max)

                b = (bell_a - self._v_min) / self._delta
                l = torch.floor(b).long()
                u = torch.ceil(b).long()

                m = torch.zeros(self._batch_size.get_value(), self._n_atoms, device=TorchUtils.get_device())
                rows = torch.arange(len(m), device=TorchUtils.get_device())
                for i in range(self._n_atoms):
                    l[:, i][(u[:, i] > 0) & (l[:, i] == u[:, i])] -= 1
                    u[:, i][(l[:, i] < (self._n_atoms - 1)) & (l[:, i] == u[:, i])] += 1

                    m[rows, l[:, i]] += p_next[:, i] * (u[:, i] - b[:, i])
                    m[rows, u[:, i]] += p_next[:, i] * (b[:, i] - l[:, i])

                kl = -torch.sum(m * torch.log(self.approximator.predict(state, action, get_distribution=True,
                                                                        **self._predict_params).clip(1e-5)), 1)
            self._replay_memory.update(kl, idxs)

            self.approximator.fit(state, action, m, weights=is_weight,
                                  get_distribution=True, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
