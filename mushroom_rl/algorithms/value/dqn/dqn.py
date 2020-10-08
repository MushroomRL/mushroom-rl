from copy import deepcopy

import numpy as np

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory, ReplayMemory


class AbstractDQN(Agent):
    def __init__(self, mdp_info, policy, approximator, approximator_params,
                 batch_size, target_update_frequency,
                 replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, fit_params=None, clip_reward=True):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            approximator_params (dict): parameters of the approximator to
                build;
            batch_size (int): the number of samples in a batch;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            replay_memory ([ReplayMemory, PrioritizedReplayMemory], None): the
                object of the replay memory to use; if None, a default replay
                memory is created;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            clip_reward (bool, True): whether to clip the reward or not.

        """
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        if replay_memory is not None:
            self._replay_memory = replay_memory
            if isinstance(replay_memory, PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = ReplayMemory(initial_replay_size,
                                               max_replay_size)
            self._fit = self._fit_standard

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)

        self._initialize_regressors(approximator, apprx_params_train,
                                    apprx_params_target)
        policy.set_q(self.approximator)

        self._add_save_attr(
            _fit_params='pickle',
            _batch_size='primitive',
            _n_approximators='primitive',
            _clip_reward='primitive',
            _target_update_frequency='primitive',
            _replay_memory='mushroom',
            _n_updates='primitive',
            approximator='mushroom',
            target_approximator='mushroom'
        )

        super().__init__(mdp_info, policy)


class DQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def _initialize_regressors(self, approximator, apprx_params_train,
                              apprx_params_target):
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        self._update_target()

    def fit(self, dataset):
        self._fit(dataset)

        self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _fit_standard(self, dataset, approximator=None):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            if approximator is None:
                self.approximator.fit(state, action, q, **self._fit_params)
            else:
                approximator.fit(state, action, q, **self._fit_params)

    def _fit_prioritized(self, dataset, approximator=None):
        self._replay_memory.add(
            dataset, np.ones(len(dataset)) * self._replay_memory.max_priority)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, idxs, is_weight = \
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            td_error = q - self.approximator.predict(state, action)

            self._replay_memory.update(td_error, idxs)

            if approximator is None:
                self.approximator.fit(state, action, q, weights=is_weight,
                                      **self._fit_params)
            else:
                approximator.fit(state, action, q, weights=is_weight,
                                 **self._fit_params)

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.set_weights(self.approximator.get_weights())

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

    def _post_load(self):
        if isinstance(self._replay_memory, PrioritizedReplayMemory):
            self._fit = self._fit_prioritized
        else:
            self._fit = self._fit_standard

        self.policy.set_q(self.approximator)
