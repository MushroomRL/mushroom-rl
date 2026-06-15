from copy import deepcopy

import torch

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import CategoricalNetwork
from mushroom_rl.utils.torch import TorchUtils

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


class AbstractCategoricalDQN(AbstractDQN):
    """
    Abstract class for DQN-based algorithms with a categorical (distributional) value function.

    """
    def __init__(self, mdp_info, policy, approximator_params, n_atoms, v_min, v_max, **params):
        """
        Constructor.

        Args:
            n_atoms (int): number of atoms;
            v_min (float): minimum value of value-function;
            v_max (float): maximum value of value-function.

        """
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        self._delta = (v_max - v_min) / (n_atoms - 1)
        self._a_values = torch.arange(v_min, v_max + eps, self._delta, device=TorchUtils.get_device())

        approximator_params['loss'] = categorical_loss

        self._add_save_attr(
            _n_atoms='primitive',
            _v_min='primitive',
            _v_max='primitive',
            _delta='primitive',
            _a_values='torch'
        )

        super().__init__(mdp_info, policy, TorchApproximator, approximator_params=approximator_params, **params)

    def _categorical_projection(self, reward, gamma, p_next):
        """
        Project the target distribution onto the fixed support of the value function.

        Args:
            reward (torch.Tensor): batch of (possibly n-step) rewards;
            gamma (torch.Tensor): per-sample discount, already zeroed on absorbing states;
            p_next (torch.Tensor): next-state probability mass over the atoms.

        Returns:
            The projected target distribution over the atoms.

        """
        gamma_z = gamma.unsqueeze(1) * self._a_values
        bell_a = (reward.unsqueeze(1) + gamma_z).clip(self._v_min, self._v_max)

        b = (bell_a - self._v_min) / self._delta
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        m = torch.zeros(len(reward), self._n_atoms, device=TorchUtils.get_device())
        rows = torch.arange(len(m), device=TorchUtils.get_device())
        for i in range(self._n_atoms):
            l[:, i][(u[:, i] > 0) & (l[:, i] == u[:, i])] -= 1
            u[:, i][(l[:, i] < (self._n_atoms - 1)) & (l[:, i] == u[:, i])] += 1

            m[rows, l[:, i]] += p_next[:, i] * (u[:, i] - b[:, i])
            m[rows, u[:, i]] += p_next[:, i] * (b[:, i] - l[:, i])

        return m


class CategoricalDQN(AbstractCategoricalDQN):
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
        approximator_params = deepcopy(approximator_params)
        approximator_params['network'] = CategoricalNetwork
        approximator_params['features_network'] = features_network
        approximator_params['n_atoms'] = n_atoms
        approximator_params['v_min'] = v_min
        approximator_params['v_max'] = v_max

        super().__init__(mdp_info, policy, approximator_params, n_atoms, v_min, v_max, **params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_ =\
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = torch.clip(reward, -1, 1)

            with torch.no_grad():
                q_next = self.target_approximator.predict(next_state, **self._predict_params)
                a_max = torch.argmax(q_next, 1)
                gamma = self.mdp_info.gamma * ~absorbing
                p_next = self.target_approximator.predict(next_state, a_max,
                                                          get_distribution=True, **self._predict_params)
                m = self._categorical_projection(reward, gamma, p_next)

            self.approximator.fit(state, action, m, get_distribution=True,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
