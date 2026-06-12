from copy import deepcopy

import numpy as np

import torch

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.approximators.parametric.networks import CategoricalNetwork

eps = torch.finfo(torch.float32).eps


def categorical_loss(input, target, reduction='sum'):
    input = input.clamp(1e-5)

    loss = -torch.sum(target * torch.log(input), 1)

    if reduction == 'sum':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError


class CategoricalDQN(AbstractDQN):
    """
    Categorical DQN algorithm.
    "A Distributional Perspective on Reinforcement Learning"
    Bellemare M. et al. 2017.

    """
    def __init__(self, mdp_info, policy, approximator_params, n_atoms, v_min,
                 v_max, **params):
        """
        Constructor.

        Args:
            n_atoms (int): number of atoms;
            v_min (float): minimum value of value-function;
            v_max (float): maximum value of value-function.

        """
        features_network = approximator_params['network']
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = CategoricalNetwork
        params['approximator_params']['features_network'] = features_network
        params['approximator_params']['n_atoms'] = n_atoms
        params['approximator_params']['v_min'] = v_min
        params['approximator_params']['v_max'] = v_max
        params['approximator_params']['loss'] = categorical_loss

        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        self._delta = (v_max - v_min) / (n_atoms - 1)
        self._a_values = np.arange(v_min, v_max + eps, self._delta)

        self._add_save_attr(
            _n_atoms='primitive',
            _v_min='primitive',
            _v_max='primitive',
            _delta='primitive',
            _a_values='numpy'
        )

        super().__init__(mdp_info, policy, NumpyTorchApproximator, **params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_ =\
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self.target_approximator.predict(next_state, **self._predict_params)
            a_max = np.argmax(q_next, 1)
            gamma = self.mdp_info.gamma * (1 - absorbing)
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

            self.approximator.fit(state, action, m, get_distribution=True,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
