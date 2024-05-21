from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.algorithms.value.dqn.categorical_dqn import categorical_loss
from mushroom_rl.algorithms.value.dqn.noisy_dqn import NoisyNetwork
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.utils.torch import TorchUtils

eps = torch.finfo(torch.float32).eps


class RainbowNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_atoms,
                 v_min, v_max, n_features, sigma_coeff, **kwargs):
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max

        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta, device=TorchUtils.get_device())

        self._pv = NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff)
        self._pa = nn.ModuleList([NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff)
                                  for _ in range(self._n_output)])

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state)

        a_pv = self._pv(features)
        a_pa = [self._pa[i](features) for i in range(self._n_output)]
        a_pa = torch.stack(a_pa, dim=1)
        a_pv = a_pv.unsqueeze(1).repeat(1, self._n_output, 1)
        mean_a_pa = a_pa.mean(1, keepdim=True).repeat(1, self._n_output, 1)
        softmax = F.softmax(a_pv + a_pa - mean_a_pa, dim=-1)

        if not get_distribution:
            q = torch.empty(softmax.shape[:-1])
            for i in range(softmax.shape[0]):
                q[i] = softmax[i] @ self._a_values

            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_atoms)

                return torch.squeeze(softmax.gather(1, action))
            else:
                return softmax


class Rainbow(AbstractDQN):
    """
    Rainbow algorithm.
    "Rainbow: Combining Improvements in Deep Reinforcement Learning".
    Hessel M. et al.. 2018.

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
        self._a_values = np.arange(v_min, v_max + eps, self._delta)
        self._n_steps_return = n_steps_return
        self._sigma_coeff = sigma_coeff

        params['replay_memory'] = {"class": PrioritizedReplayMemory,
                                   "params": dict(alpha=alpha_coeff, beta=beta)}

        super().__init__(mdp_info, policy, NumpyTorchApproximator, **params)

        self._add_save_attr(
            _n_atoms='primitive',
            _v_min='primitive',
            _v_max='primitive',
            _delta='primitive',
            _a_values='numpy',
            _n_steps_return='primitive',
            _sigma_coeff='primitive'
        )

    def fit(self, dataset):
        self._replay_memory.add(dataset, np.ones(len(dataset)) * self._replay_memory.max_priority,
                                n_steps_return=self._n_steps_return, gamma=self.mdp_info.gamma)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, idxs, is_weight = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self.approximator.predict(next_state, **self._predict_params)
            a_max = np.argmax(q_next, axis=1)
            gamma = self.mdp_info.gamma ** self._n_steps_return * (1 - absorbing)
            p_next = self.target_approximator.predict(next_state, a_max,
                                                      get_distribution=True, **self._predict_params)
            gamma_z = gamma.reshape(-1, 1) * np.expand_dims(
                self._a_values, 0).repeat(len(gamma), 0)
            bell_a = (reward.reshape(-1, 1) + gamma_z).clip(self._v_min,
                                                            self._v_max)

            b = (bell_a - self._v_min) / self._delta
            l = np.floor(b).astype(int)
            u = np.ceil(b).astype(int)

            m = np.zeros((self._batch_size.get_value(), self._n_atoms))
            for i in range(self._n_atoms):
                l[:, i][(u[:, i] > 0) * (l[:, i] == u[:, i])] -= 1
                u[:, i][(l[:, i] < (self._n_atoms - 1)) * (l[:, i] == u[:, i])] += 1

                m[np.arange(len(m)), l[:, i]] += p_next[:, i] * (u[:, i] - b[:, i])
                m[np.arange(len(m)), u[:, i]] += p_next[:, i] * (b[:, i] - l[:, i])

            kl = -np.sum(m * np.log(self.approximator.predict(state, action, get_distribution=True,
                                                              **self._predict_params).clip(1e-5)), 1)
            self._replay_memory.update(kl, idxs)

            self.approximator.fit(state, action, m, weights=is_weight,
                                  get_distribution=True, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
