from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.approximators.parametric.pytorch_network import *
from mushroom.approximators.regressor import Ensemble, Regressor
from mushroom.utils.replay_memory import ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size,
                 approximator_params, target_update_frequency,
                 fit_params=None, n_approximators=1, clip_reward=True):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            approximator_params (dict): parameters of the approximator to
                build;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            n_approximators (int, 1): the number of approximator to use in
                ``AverageDQN``;
            clip_reward (bool, True): whether to clip the reward or not.

        """
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             n_models=self._n_approximators,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        if self._n_approximators == 1:
            self.target_approximator.model.set_weights(
                self.approximator.model.get_weights())
        else:
            for i in range(self._n_approximators):
                self.target_approximator.model[i].set_weights(
                    self.approximator.model.get_weights())

        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self.approximator.fit(state, action, q, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Maximum action-value for each state in ``next_state``.

        """
        q = self.target_approximator.predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

    def draw_action(self, state):
        action = super(DQN, self).draw_action(np.array(state))

        return action


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.

    """
    def _next_q(self, next_state, absorbing):
        q = self.approximator.predict(next_state)
        max_a = np.argmax(q, axis=1)

        double_q = self.target_approximator.predict(next_state, max_a)
        if np.any(absorbing):
            double_q *= 1 - absorbing

        return double_q


class AveragedDQN(DQN):
    """
    Averaged-DQN algorithm.
    "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement
    Learning". Anschel O. et al.. 2017.

    """
    def __init__(self, approximator, policy, mdp_info, **params):
        super().__init__(approximator, policy, mdp_info, **params)

        self._n_fitted_target_models = 1

        assert isinstance(self.target_approximator.model, Ensemble)

    def _update_target(self):
        idx = self._n_updates // self._target_update_frequency\
              % self._n_approximators
        self.target_approximator.model[idx].set_weights(
            self.approximator.model.get_weights())

        if self._n_fitted_target_models < self._n_approximators:
            self._n_fitted_target_models += 1

    def _next_q(self, next_state, absorbing):
        q = list()
        for idx in range(self._n_fitted_target_models):
            q.append(self.target_approximator.predict(next_state, idx=idx))
        q = np.mean(q, axis=0)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)


class CategoricalNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_atoms,
                 v_min, v_max, n_features, use_cuda, **kwargs):
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max

        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + delta, delta)
        if use_cuda:
            self._a_values = self._a_values.cuda()

        self._p = nn.ModuleList(
            [nn.Linear(n_features, n_atoms) for _ in range(self._n_output)])

        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._p[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state, action)

        a_p = [F.softmax(self._p[i](features), -1) for i in range(self._n_output)]
        a_p = torch.stack(a_p, dim=1)

        if not get_distribution:
            q = torch.empty(a_p.shape[:-1])
            for i in range(a_p.shape[0]):
                q[i] = a_p[i] @ self._a_values

            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_atoms)

                return torch.squeeze(a_p.gather(1, action))
            else:
                return a_p


class CategoricalDQN(DQN):
    """
    Categorical DQN algorithm.
    "A Distributional Perspective on Reinforcement Learning".
    Bellemare M. et al.. 2017.

    """
    def __init__(self, policy, mdp_info, n_atoms, v_min, v_max,
                 approximator_params, **params):
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

        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        self._delta = (v_max - v_min) / (n_atoms - 1)
        self._a_values = np.arange(v_min, v_max + self._delta, self._delta)

        super().__init__(PyTorchApproximator, policy, mdp_info, **params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self.target_approximator.predict(next_state)
            a_max = np.argmax(q_next, 1)
            gamma = self.mdp_info.gamma * (1 - absorbing)
            p_next = self.target_approximator.predict(next_state, a_max,
                                                      get_distribution=True)
            gamma_z = gamma.reshape(-1, 1) * np.expand_dims(
                self._a_values, 0).repeat(len(gamma), 0)
            bell_a = (reward.reshape(-1, 1) + gamma_z).clip(self._v_min,
                                                            self._v_max)

            b = (bell_a - self._v_min) / self._delta
            l = np.floor(b).astype(np.int)
            u = np.ceil(b).astype(np.int)

            m = np.zeros((self._batch_size, self._n_atoms))
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

    def draw_action(self, state):
        action = super(DQN, self).draw_action(np.array(state))

        return action
