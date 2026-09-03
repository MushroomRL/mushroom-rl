from copy import deepcopy

import torch

from mushroom_rl.core import Agent
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory, ReplayMemory
from mushroom_rl.rl_utils.parameters import Parameter


class AbstractDQN(Agent):
    """
    Abstract class for every DQN-based approach.

    """

    def __init__(self, mdp_info, policy, approximator, approximator_params, batch_size, target_update_frequency,
                 replay_memory=None, initial_replay_size=500, max_replay_size=5000, fit_params=None,
                 predict_params=None, clip_reward=False, history_length=1):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the Q-function;
            approximator_params (dict): parameters of the approximator to build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            target_update_frequency (int): the number of samples collected between each update of the target network;
            replay_memory ([dict, ReplayMemory, PrioritizedReplayMemory, None]): if a dict, must have keys 'class' and
                'params' and the class will be instantiated with mdp_info and agent_info; if already an instance, it is
                used directly; if None a default ReplayMemory is created;
            initial_replay_size (int): the number of samples to collect before starting the learning;
            max_replay_size (int): the maximum number of samples in the replay memory;
            fit_params (dict, None): parameters of the fitting algorithm of the approximator;
            predict_params (dict, None): parameters for the prediction with the approximator;
            clip_reward (bool, False): whether to clip the reward or not;
            history_length (int, 1): number of consecutive observation stacked as policy input.

        """
        super().__init__(mdp_info, policy, backend='torch', history_length=history_length)

        assert policy._backend == self._agent_backend, \
            f"The policy uses the '{policy._backend.get_backend_name()}' backend, but DQN-based agents " \
            f"run on the '{self._agent_backend.get_backend_name()}' backend. Build the policy with " \
            f"backend='{self._agent_backend.get_backend_name()}', e.g. " \
            f"EpsGreedy(epsilon, backend='{self._agent_backend.get_backend_name()}')."

        self._fit_params = dict()
        self._predict_params = dict()

        if fit_params is not None:
            self._fit_params.update(fit_params)

        if predict_params is not None:
            self._predict_params.update(predict_params)

        self._batch_size = Parameter.make(batch_size, backend='torch')
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        if replay_memory is not None:
            if isinstance(replay_memory, dict):
                self._replay_memory = replay_memory["class"](
                    mdp_info, self.info, initial_replay_size, max_replay_size,
                    history_manager=self.history_manager, **replay_memory["params"])
            else:
                self._replay_memory = replay_memory

            if isinstance(self._replay_memory, PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = ReplayMemory(
                mdp_info, self.info, initial_replay_size, max_replay_size,
                history_manager=self.history_manager)
            self._fit = self._fit_standard

        self._n_updates = 0

        self._initialize_regressors(approximator, approximator_params)

        policy.set_q(self.approximator)

        self._add_save_attr(
            _fit_params='pickle',
            _predict_params='pickle',
            _batch_size='mushroom',
            _clip_reward='primitive',
            _target_update_frequency='primitive',
            _replay_memory='mushroom',
            _n_updates='primitive',
            approximator='mushroom',
            target_approximator='mushroom'
        )

        self._add_logger_attr('approximator', group='critic')

    def fit(self, dataset):
        self._history_manager.update_preprocessors(dataset)
        self._fit(dataset)

        if self._replay_memory.initialized:
            self._n_updates += 1
            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

        if self._logger:
            self._logger.advance_step()

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_ = self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = torch.clip(reward, -1, 1)

            with torch.no_grad():
                target = self._compute_target(reward, next_state, absorbing)

            self.approximator.fit(state, action, target, **self._fit_params)

    def _fit_prioritized(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_, idxs, is_weight = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = torch.clip(reward, -1, 1)

            with torch.no_grad():
                target = self._compute_target(reward, next_state, absorbing)
                priority = self._compute_priority(state, action, target)

            self._replay_memory.update(priority, idxs)

            self.approximator.fit(state, action, target, weights=is_weight, **self._fit_params)

    def _initialize_regressors(self, approximator, approximator_params):
        """
        Build the online and target approximators.

        Args:
            approximator (class): the approximator class to instantiate;
            approximator_params (dict): parameters of the approximators to build. Each approximator is built from
                its own deep copy, so the two never share a nested object.

        """
        self.approximator = approximator(**deepcopy(approximator_params))
        self.target_approximator = approximator(**deepcopy(approximator_params))
        self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.set_weights(self.approximator.get_weights())

    def _compute_target(self, reward, next_state, absorbing):
        """
        Compute the regression target the approximator is fitted on.

        Args:
            reward (torch.Tensor): batch of rewards;
            next_state (torch.Tensor): the states where next action has to be
                evaluated;
            absorbing (torch.Tensor): the absorbing flag for the states in
                ``next_state``.

        Returns:
            The bootstrapped target of each sample of the batch.

        """
        q_next = self._next_q(next_state, absorbing)
        return reward + self.mdp_info.gamma * q_next

    def _compute_priority(self, state, action, target):
        """
        Compute the priority of the batch samples for a ``PrioritizedReplayMemory``.

        Args:
            state (torch.Tensor): batch of states;
            action (torch.Tensor): batch of actions;
            target (torch.Tensor): batch of regression targets.

        Returns:
            The priority of each sample of the batch.

        """
        return target - self.approximator.predict(state, action, **self._predict_params)

    def _next_q(self, next_state, absorbing):
        """
        Compute the value of the next states, bootstrapped by :meth:`_compute_target`.

        Args:
            next_state (torch.Tensor): the states where next action has to be
                evaluated;
            absorbing (torch.Tensor): the absorbing flag for the states in
                ``next_state``.

        Returns:
            The value of each state in ``next_state``, zero where the state is absorbing.

        """
        raise NotImplementedError

    def _post_load(self):
        if isinstance(self._replay_memory, PrioritizedReplayMemory):
            self._fit = self._fit_prioritized
        else:
            self._fit = self._fit_standard

        self.policy.set_q(self.approximator)
