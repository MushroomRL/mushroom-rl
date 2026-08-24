from copy import deepcopy

import torch

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import CategoricalNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

eps = torch.finfo(torch.float32).eps


def categorical_loss(input, target, reduction='mean'):
    input = input.clamp(1e-5)

    loss = -torch.sum(target * torch.log(input), 1)

    if reduction == 'mean':
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

        approximator_params = dict(approximator_params)
        approximator_params.update(loss=categorical_loss)

        self._add_save_attr(
            _n_atoms='primitive',
            _v_min='primitive',
            _v_max='primitive',
            _delta='primitive',
            _a_values='torch'
        )

        super().__init__(mdp_info, policy, TorchApproximator, approximator_params=approximator_params, **params)

        self._fit_params['get_distribution'] = True

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
        low_b = torch.floor(b).long()
        upp_b = torch.ceil(b).long()

        m = torch.zeros(len(reward), self._n_atoms, device=TorchUtils.get_device())
        rows = torch.arange(len(m), device=TorchUtils.get_device())
        for i in range(self._n_atoms):
            low_b[:, i][(upp_b[:, i] > 0) & (low_b[:, i] == upp_b[:, i])] -= 1
            upp_b[:, i][(low_b[:, i] < (self._n_atoms - 1)) & (low_b[:, i] == upp_b[:, i])] += 1

            m[rows, low_b[:, i]] += p_next[:, i] * (upp_b[:, i] - b[:, i])
            m[rows, upp_b[:, i]] += p_next[:, i] * (b[:, i] - low_b[:, i])

        return m

    def _compute_priority(self, state, action, target):
        p = self.approximator.predict(state, action, get_distribution=True, **self._predict_params)

        return categorical_loss(p, target, reduction='none')


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

    def _compute_target(self, reward, next_state, absorbing):
        q_next = self.target_approximator.predict(next_state, **self._predict_params)
        a_max = torch.argmax(q_next, 1).unsqueeze(1)
        gamma = self.mdp_info.gamma * ~absorbing
        p_next = self.target_approximator.predict(next_state, a_max, get_distribution=True,
                                                  **self._predict_params)

        return self._categorical_projection(reward, gamma, p_next)
