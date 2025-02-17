import numpy as np
import math

from collections import defaultdict

import torch

from mushroom_rl.core.serialization import Serializable
from .array_backend import ArrayBackend
from .extra_info import ExtraInfo

from ._impl import *

from mushroom_rl.utils.episodes import split_episodes, unsplit_episodes

class DatasetInfo(Serializable):
    def __init__(self, backend, device, horizon, gamma, state_shape, state_dtype, action_shape, action_dtype,
                 policy_state_shape, n_envs=1):
        assert backend == "torch" or device is None

        self.backend = backend
        self.device = device
        self.horizon = horizon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.policy_state_shape = policy_state_shape
        self.n_envs = n_envs

        super().__init__()

        self._add_save_attr(
            backend='primitive',
            gamma='primitive',
            horizon='primitive',
            state_shape='primitive',
            state_dtype='primitive',
            action_shape='primitive',
            action_dtype='primitive',
            policy_state_shape='primitive',
            n_envs='primitive'
        )

    @property
    def is_agent_stateful(self):
        return self.policy_state_shape is not None

    @staticmethod
    def create_dataset_info(mdp_info, agent_info, n_envs=1, device=None):
        backend = mdp_info.backend
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = mdp_info.observation_space.data_type
        action_shape = mdp_info.action_space.shape
        action_dtype = mdp_info.action_space.data_type
        policy_state_shape = agent_info.policy_state_shape

        return DatasetInfo(backend, device, horizon, gamma, state_shape, state_dtype,
                           action_shape, action_dtype, policy_state_shape, n_envs)

    @staticmethod
    def create_replay_memory_info(mdp_info, agent_info, device=None):
        backend = agent_info.backend
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = mdp_info.observation_space.data_type  # FIXME: this may cause issues, needs fix
        action_shape = mdp_info.action_space.shape
        action_dtype = mdp_info.action_space.data_type  # FIXME: this may cause issues, needs fix
        policy_state_shape = agent_info.policy_state_shape

        return DatasetInfo(backend, device, horizon, gamma, state_shape, state_dtype,
                           action_shape, action_dtype, policy_state_shape)


class Dataset(Serializable):
    def __init__(self, dataset_info, n_steps=None, n_episodes=None, core_counts_episodes=False):
        assert (n_steps is not None and n_episodes is None) or (n_steps is None and n_episodes is not None)

        self._array_backend = ArrayBackend.get_array_backend(dataset_info.backend)

        if n_steps is not None:
            n_samples = n_steps
        else:
            horizon = dataset_info.horizon
            assert np.isfinite(horizon)

            n_samples = horizon * n_episodes

        if dataset_info.n_envs == 1:
            base_shape = (n_samples,)
            mask_shape = None
        elif n_episodes:
            horizon = dataset_info.horizon
            x = math.ceil(n_episodes / dataset_info.n_envs)
            base_shape = (x * horizon, min(n_episodes, dataset_info.n_envs))
            mask_shape = base_shape
        elif core_counts_episodes:
            base_shape = (math.ceil(n_samples / dataset_info.n_envs) + 1 + dataset_info.horizon, dataset_info.n_envs)
            mask_shape = base_shape
        else:
            base_shape = (math.ceil(n_samples / dataset_info.n_envs) + 1, dataset_info.n_envs)
            mask_shape = base_shape

        state_shape = base_shape + dataset_info.state_shape
        action_shape = base_shape + dataset_info.action_shape
        reward_shape = base_shape

        if dataset_info.is_agent_stateful:
            policy_state_shape = base_shape + dataset_info.policy_state_shape
        else:
            policy_state_shape = None

        self._info = ExtraInfo(min(n_episodes, dataset_info.n_envs) if n_episodes else dataset_info.n_envs, dataset_info.backend, dataset_info.device)
        self._episode_info = ExtraInfo(min(n_episodes, dataset_info.n_envs) if n_episodes else dataset_info.n_envs, dataset_info.backend, dataset_info.device)
        self._theta_list = list()

        if dataset_info.backend == 'numpy':
            self._data = NumpyDataset(dataset_info.state_dtype, state_shape,
                                      dataset_info.action_dtype, action_shape,
                                      reward_shape, base_shape,
                                      policy_state_shape, mask_shape)
        elif dataset_info.backend == 'torch':
            self._data = TorchDataset(dataset_info.state_dtype, state_shape,
                                      dataset_info.action_dtype, action_shape, reward_shape, base_shape,
                                      policy_state_shape, mask_shape, device=dataset_info.device)
        else:
            self._data = ListDataset(policy_state_shape is not None, mask_shape is not None)

        self._dataset_info = dataset_info

        super().__init__()

        self._add_all_save_attr()

    @classmethod
    def generate(cls, mdp_info, agent_info, n_steps=None, n_episodes=None, n_envs=1, core_counts_episodes=False):
        dataset_info = DatasetInfo.create_dataset_info(mdp_info, agent_info, n_envs)

        return cls(dataset_info, n_steps, n_episodes, core_counts_episodes)

    @classmethod
    def create_raw_instance(cls, dataset=None):
        """
        Creates an empty instance of the Dataset and populates essential data structures

        Args:
            dataset (Dataset, None): a template dataset to be used to create the new instance.

        Returns:
            A new empty instance of the dataset.

        """
        new_dataset = cls.__new__(cls)

        if dataset is not None:
            new_dataset._array_backend = dataset._array_backend
            new_dataset._dataset_info = dataset._dataset_info
        else:
            new_dataset._dataset_info = None

        new_dataset._info = None
        new_dataset._episode_info = None
        new_dataset._data = None
        new_dataset._theta_list = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_state=None, policy_next_state=None, info=None, episode_info=None, theta_list=None,
                   horizon=None, gamma=0.99, backend='numpy', device=None):
        """
        Creates a dataset of transitions from the provided arrays.

        Args:
            states (array): array of states;
            actions (array): array of actions;
            rewards (array): array of rewards;
            next_states (array): array of next_states;
            absorbings (array): array of absorbing flags;
            lasts (array): array of last flags;
            policy_state (array, None): array of policy internal states;
            policy_next_state (array, None): array of next policy internal states;
            info (dict, None): dictiornay of step info;
            episode_info (dict, None): dictiornary of episode info;
            theta_list (list, None): list of policy parameters;
            horizon (int, None): horizon of the mdp;
            gamma (float, 0.99): discount factor;
            backend (str, 'numpy'): backend to be used by the dataset.

        Returns:
            The list of transitions.

        """
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(absorbings) == len(lasts)

        if policy_state is not None:
            assert len(states) == len(policy_state) == len(policy_next_state)

        dataset = cls.create_raw_instance()

        if info is None:
            dataset._info = ExtraInfo(1, backend)
        else:
            dataset._info = info.copy()

        if episode_info is None:
            dataset._episode_info = ExtraInfo(1, backend)
        else:
            dataset._episode_info = episode_info.copy()

        if theta_list is None:
            dataset._theta_list = list()
        else:
            dataset._theta_list = theta_list

        dataset._array_backend = ArrayBackend.get_array_backend(backend)
        if backend == 'numpy':
            dataset._data = NumpyDataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                                    policy_state, policy_next_state)
        elif backend == 'torch':
            dataset._data = TorchDataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                                    policy_state, policy_next_state)
        else:
            dataset._data = ListDataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                                   policy_state, policy_next_state)

        state_shape = states.shape[1:]
        action_shape = actions.shape[1:]
        policy_state_shape = None if policy_state is None else policy_state.shape[1:]

        dataset._dataset_info = DatasetInfo(backend, device, horizon, gamma, state_shape, states.dtype,
                                            action_shape, actions.dtype, policy_state_shape)

        return dataset

    def append(self, step, info):
        self._data.append(*step)
        self._info.append(info)

    def append_episode_info(self, info):
        self._append_info(self._episode_info, info)

    def append_theta(self, theta):
        self._theta_list.append(theta)

    def get_info(self, field, index=None):
        if index is None:
            return self._info[field]
        else:
            return self._info[field][index]

    def clear(self):
        self._episode_info.clear()
        self._theta_list = list()
        self._info.clear()

        self._data.clear()

    def get_view(self, index, copy=False):
        dataset = self.create_raw_instance(dataset=self)

        dataset._info = self._info.get_view(index, copy)
        dataset._episode_info = self._episode_info.get_view(index, copy)
        dataset._data = self._data.get_view(index, copy)

        return dataset

    def item(self):
        assert len(self) == 1
        return self[0]

    def __getitem__(self, index):
        if isinstance(index, (slice, np.ndarray)) or isinstance(index, (slice, torch.Tensor)):
            return self.get_view(index)
        elif isinstance(index, int) and index < len(self._data):
            return self._data[index]
        else:
            raise IndexError

    def __add__(self, other):
        result = self.create_raw_instance(dataset=self)

        result._info = self._info + other._info
        result._episode_info = self._episode_info + other._episode_info
        result._theta_list = self._theta_list + other._theta_list
        result._data = self._data + other._data

        return result

    def __len__(self):
        return len(self._data)

    @property
    def state(self):
        return self._data.state

    @property
    def action(self):
        return self._data.action

    @property
    def reward(self):
        return self._data.reward

    @property
    def next_state(self):
        return self._data.next_state

    @property
    def absorbing(self):
        return self._data.absorbing

    @property
    def last(self):
        return self._data.last

    @property
    def policy_state(self):
        return self._data.policy_state

    @property
    def policy_next_state(self):
        return self._data.policy_next_state

    @property
    def info(self):
        return self._info

    @property
    def episode_info(self):
        return self._episode_info

    @property
    def theta_list(self):
        return self._theta_list

    @property
    def episodes_length(self):
        """
        Compute the length of each episode in the dataset.

        Returns:
            A list of length of each episode in the dataset.

        """
        lengths = list()
        l = 0
        for sample in self:
            l += 1
            if sample[-1] == 1:
                lengths.append(l)
                l = 0

        return self._array_backend.from_list(lengths)

    @property
    def n_episodes(self):
        return self._data.n_episodes

    @property
    def undiscounted_return(self):
        return self.compute_J()

    @property
    def discounted_return(self):
        return self.compute_J(self._dataset_info.gamma)

    @property
    def array_backend(self):
        return self._array_backend

    @property
    def is_stateful(self):
        return self._data.is_stateful

    def parse(self, to=None):
        """
        Return the dataset as set of arrays.
        Args:
            to (str, None):  the backend to be used for the returned arrays. By default, the dataset backend is used.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, next state, absorbing and last

        """
        if to is None:
            to = self._array_backend.get_backend_name()
        return self._convert(self.state, self.action, self.reward, self.next_state, self.absorbing, self.last, to=to)

    def parse_policy_state(self, to=None):
        """
        Return the dataset as set of arrays.

        Args:
            to (str, None):  the backend to be used for the returned arrays. By default, the dataset backend is used.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, next state, absorbing and last

        """
        if to is None:
            to = self._array_backend.get_backend_name()
        return self._convert(self.policy_state, self.policy_next_state, to=to)

    def select_first_episodes(self, n_episodes):
        """
        Return the first ``n_episodes`` episodes in the provided dataset.

        Args:
            n_episodes (int): the number of episodes to pick from the dataset;

        Returns:
            A subset of the dataset containing the first ``n_episodes`` episodes.

        """
        assert n_episodes > 0, 'Number of episodes must be greater than zero.'

        last_idxs = np.argwhere(self.last).ravel()
        return self[:last_idxs[n_episodes - 1] + 1]

    def select_random_samples(self, n_samples):
        """
        Return the randomly picked desired number of samples in the provided
        dataset.

        Args:
            n_samples (int): the number of samples to pick from the dataset.

        Returns:
            A subset of the dataset containing randomly picked ``n_samples``
            samples.

        """
        assert n_samples >= 0, 'Number of samples must be greater than or equal to zero.'

        if n_samples == 0:
            return np.array([[]])

        idxs = np.random.randint(len(self), size=n_samples)

        return self[idxs]

    def get_init_states(self):
        """
        Get the initial states of a dataset

        Returns:
            An array of initial states of the considered dataset.

        """
        pick = True
        x_0 = list()
        for step in self:
            if pick:
                x_0.append(step[0])
            pick = step[-1]
        return self._array_backend.from_list(x_0)

    def compute_J(self, gamma=1.):
        """
        Compute the cumulative discounted reward of each episode in the dataset.

        Args:
            gamma (float, 1.): discount factor.

        Returns:
            The cumulative discounted reward of each episode in the dataset.

        """
        r_ep = split_episodes(self.last, self.reward)

        if len(r_ep.shape) == 1:
            r_ep = self._array_backend.expand_dims(r_ep, 0)
        if self._dataset_info.backend == 'torch':
            js = self._array_backend.zeros(r_ep.shape[0], dtype=r_ep.dtype, device=r_ep.device)
        else:
            js = self._array_backend.zeros(r_ep.shape[0], dtype=r_ep.dtype)

        for k in range(r_ep.shape[1]):
            js += gamma ** k * r_ep[..., k]

        return js

    def compute_metrics(self, gamma=1.):
        """
        Compute the metrics of each complete episode in the dataset.

        Args:
            gamma (float, 1.): the discount factor.

        Returns:
            The minimum score reached in an episode,
            the maximum score reached in an episode,
            the mean score reached,
            the median score reached,
            the number of completed episodes.

            If no episode has been completed, it returns 0 for all values.

        """
        i = 0
        for i in reversed(range(len(self))):
            if self.last[i]:
                i += 1
                break

        dataset = self[:i]

        if len(dataset) > 0:
            J = dataset.compute_J(gamma)
            median = self._array_backend.median(J)
            return J.min(), J.max(), J.mean(), median, len(J)
        else:
            return 0, 0, 0, 0, 0

    def _convert(self, *arrays, to='numpy'):
        if to == 'numpy':
            return self._array_backend.arrays_to_numpy(*arrays)
        elif to == 'torch':
            return self._array_backend.arrays_to_torch(*arrays)
        else:
            return NotImplementedError

    def _add_all_save_attr(self):
        self._add_save_attr(
            _info='mushroom',
            _episode_info='mushroom',
            _theta_list='pickle',
            _data='mushroom',
            _array_backend='primitive',
            _dataset_info='mushroom'
        )

    @staticmethod
    def _append_info(info, step_info):
        for key, value in step_info.items():
            info[key].append(value)

    @staticmethod
    def _merge_info(info, other_info):
        new_info = defaultdict(list)
        for key in info.keys():
            new_info[key] = info[key] + other_info[key]
        return new_info


class VectorizedDataset(Dataset):
    def __init__(self, dataset_info, n_steps=None, n_episodes=None, core_counts_episodes=False):
        super().__init__(dataset_info, n_steps, n_episodes, core_counts_episodes)

        self._initialize_theta_list(self._dataset_info.n_envs)

    def append(self, step, info):
        raise RuntimeError("Trying to use append on a vectorized dataset")

    def append_vectorized(self, step, info, mask):
        self._data.append(*step, mask=mask)
        self._info.append(info)

    def append_theta_vectorized(self, theta, mask):
        for i in range(len(theta)):
            if mask[i]:
                self._theta_list[i].append(theta[i])

    def clear(self, n_steps_per_fit=None):
        n_envs = len(self._theta_list)
        n_carry_forward_steps = 0

        residual_data = None
        if n_steps_per_fit is not None:
            n_steps_dataset = self._data.mask.sum().item()

            if n_steps_dataset > n_steps_per_fit:
                n_extra_steps = n_steps_dataset - n_steps_per_fit
                n_parallel_steps = int(np.ceil(n_extra_steps / self._dataset_info.n_envs))
                view_size = slice(-n_parallel_steps, None)
                residual_data = self._data.get_view(view_size, copy=True)
                mask = residual_data.mask
                original_shape = mask.shape
                mask = mask.flatten()
                true_indices = self._array_backend.where(mask)[0]
                mask[true_indices[n_extra_steps:]] = False
                residual_data.mask = mask.reshape(original_shape)

                residual_info = self._info.get_view(view_size, copy=True)
                residual_episode_info = self._episode_info.get_view(view_size, copy=True)

                n_carry_forward_steps = mask.sum()

        super().clear()
        self._initialize_theta_list(n_envs)

        if n_steps_per_fit is not None and residual_data is not None:
            self._data = residual_data
            self._info = residual_info
            self._episode_info = residual_episode_info

        return n_carry_forward_steps

    def flatten(self, n_steps_per_fit=None):
        if len(self) == 0:
            return None

        states = self._array_backend.pack_padded_sequence(self._data.state, self._data.mask)
        actions = self._array_backend.pack_padded_sequence(self._data.action, self._data.mask)
        rewards = self._array_backend.pack_padded_sequence(self._data.reward, self._data.mask)
        next_states = self._array_backend.pack_padded_sequence(self._data.next_state, self._data.mask)
        absorbings = self._array_backend.pack_padded_sequence(self._data.absorbing, self._data.mask)

        last_padded = self._data.last
        last_padded[-1, :] = True
        lasts = self._array_backend.pack_padded_sequence(last_padded, self._data.mask)

        policy_state = None
        policy_next_state = None

        if self._data.is_stateful:
            policy_state = self._array_backend.pack_padded_sequence(self._data.policy_state, self._data.mask)
            policy_next_state = self._array_backend.pack_padded_sequence(self._data.policy_next_state, self._data.mask)

        if n_steps_per_fit is not None:
            states = states[:n_steps_per_fit]
            actions = actions[:n_steps_per_fit]
            rewards = rewards[:n_steps_per_fit]
            next_states = next_states[:n_steps_per_fit]
            absorbings = absorbings[:n_steps_per_fit]
            lasts = lasts[:n_steps_per_fit]

            if self._data.is_stateful:
                policy_state = policy_state[:n_steps_per_fit]
                policy_next_state = policy_next_state[:n_steps_per_fit]

        flat_theta_list = self._flatten_theta_list()

        flat_info = self._info.flatten(self.mask)
        flat_episode_info = self._episode_info.flatten(self.mask)

        return Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                  policy_state=policy_state, policy_next_state=policy_next_state,
                                  info=flat_info, episode_info=flat_episode_info, theta_list=flat_theta_list,
                                  horizon=self._dataset_info.horizon, gamma=self._dataset_info.gamma,
                                  backend=self._array_backend.get_backend_name())

    def _flatten_theta_list(self):
        flat_theta_list = list()

        for env_theta_list in self._theta_list:
            flat_theta_list += env_theta_list

        return flat_theta_list

    def _initialize_theta_list(self, n_envs):
        self._theta_list = list()
        for i in range(n_envs):
            self._theta_list.append(list())

    @property
    def mask(self):
        return self._data.mask
