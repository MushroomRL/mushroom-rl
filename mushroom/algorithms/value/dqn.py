from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor
from mushroom.utils.replay_memory import Buffer, ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 train_frequency, target_update_frequency, initial_replay_size,
                 max_replay_size, fit_params=None, approximator_params=None,
                 n_approximators=1, history_length=1, clip_reward=True,
                 max_no_op_actions=0, no_op_action_value=0):
        """
        Constructor.

        Args:
            batch_size (int): the number of samples in a batch;
            train_frequency (int): the number of samples collected between each
                gradient descent step of the online network;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            approximator_params (dict, None): parameters of the approximator to
                build;
            n_approximators (int, 1): the number of approximator to use in
                ``AverageDQN``;
            history_length (int, 1): the number of samples composing a state;
            clip_reward (bool, True): whether to clip the reward or not;
            max_no_op_actions (int, 0): maximum number of no-op actions that
                can be sampled;
            no_op_action_value (int, 0): value of the no-op action.

        """
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency / train_frequency
        self._max_no_op_actions = max_no_op_actions
        self._no_op_action_value = no_op_action_value

        self._replay_memory = ReplayMemory(mdp_info, initial_replay_size,
                                           max_replay_size, history_length)
        self._buffer = Buffer(size=history_length)

        self._n_updates = 0
        self._episode_steps = 0
        self._no_op_actions = None

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['name'] = 'target'
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

        super(DQN, self).__init__(policy, mdp_info)

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
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
            self.policy.update()
        else:
            extended_state = self._buffer.get()

            action = super(DQN, self).draw_action(extended_state)

        self._episode_steps += 1

        return action

    def episode_start(self):
        if self._max_no_op_actions == 0:
            self._no_op_actions = 0
        else:
            self._no_op_actions = np.random.randint(
                self._buffer.size, self._max_no_op_actions + 1)
        self._episode_steps = 0


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
        super(AveragedDQN, self).__init__(approximator, policy, mdp_info,
                                          **params)

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
