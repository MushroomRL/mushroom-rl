import numpy as np
import math

from collections import defaultdict
from enum import IntEnum

import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.extra_info import ExtraInfo

from ._impl import *

from mushroom_rl.utils.episodes import split_episodes


class DatasetInfo(MushroomObject):
    """
    Static information needed to build a :class:`Dataset`. A dataset keeps its data in two backend-aware groups:
    the environment data (state, action, reward, next state, absorbing and last flags) and the agent data (the
    policy state). This class stores the array backend and device of each group, together with the shapes and
    dtypes of the states and actions, the horizon, the discount factor and the number of parallel environments.
    Build it with the :meth:`create_dataset_info` (on-policy collection) or :meth:`create_replay_memory_info`
    (replay buffer) factories.

    """
    def __init__(self, env_backend, agent_backend, env_device, agent_device, horizon, gamma, state_shape, state_dtype,
                 action_shape, action_dtype, policy_state_shape, n_envs=1):
        """
        Constructor.

        Args:
            env_backend (str): array backend of the environment data (``'numpy'``, ``'torch'`` or ``'list'``);
            agent_backend (str): array backend of the agent (policy state) data;
            env_device (str, None): device of the environment data, only allowed with the torch backend;
            agent_device (str, None): device of the agent data, only allowed with the torch backend;
            horizon (int): horizon of the MDP;
            gamma (float): discount factor;
            state_shape (tuple): shape of a single state;
            state_dtype: data type of the states;
            action_shape (tuple): shape of a single action;
            action_dtype: data type of the actions;
            policy_state_shape (tuple, None): shape of the policy state, or ``None`` if the agent is stateless;
            n_envs (int, 1): number of parallel environments.

        """
        assert env_backend == "torch" or env_device is None
        assert agent_backend == "torch" or agent_device is None

        self.env_backend = env_backend
        self.agent_backend = agent_backend
        self.env_device = env_device
        self.agent_device = agent_device
        self.horizon = horizon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.policy_state_shape = policy_state_shape
        self.n_envs = n_envs

        self._add_save_attr(
            env_backend='primitive',
            agent_backend='primitive',
            env_device='primitive',
            agent_device='primitive',
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
    def env_array_backend(self):
        """
        The :class:`ArrayBackend` of the environment data.

        """
        return ArrayBackend.get_array_backend(self.env_backend)

    @property
    def agent_array_backend(self):
        """
        The :class:`ArrayBackend` of the agent (policy state) data.

        """
        return ArrayBackend.get_array_backend(self.agent_backend)

    @staticmethod
    def create_dataset_info(mdp_info, agent_info, n_envs=1, device=None):
        """
        Build the dataset info for on-policy collection: the environment data uses ``mdp_info.backend`` (forced
        to ``'list'`` for infinite-horizon MDPs) and the agent data uses ``agent_info.backend``.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            n_envs (int, 1): number of parallel environments;
            device (str, None): torch device used by the torch-backed data groups.

        Returns:
            The dataset info.

        """
        env_backend = mdp_info.backend
        if not np.isfinite(mdp_info.horizon):
            assert env_backend != 'torch', "Infinite-horizon collection is not supported for the torch backend."
            env_backend = 'list'
        env_device = device if env_backend == 'torch' else None
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = mdp_info.observation_space.data_type
        action_shape = mdp_info.action_space.shape
        action_dtype = mdp_info.action_space.data_type
        policy_state_shape = agent_info.policy_state_shape
        agent_device = device if agent_info.backend == 'torch' else None

        return DatasetInfo(env_backend, agent_info.backend, env_device, agent_device, horizon, gamma,
                           state_shape, state_dtype, action_shape, action_dtype, policy_state_shape, n_envs=n_envs)

    @staticmethod
    def create_replay_memory_info(mdp_info, agent_info, store_policy_state=True, device=None):
        """
        Build the dataset info for a replay memory: the whole buffer (both the transition data and the policy
        state) lives in the agent backend, so the environment and agent backends/devices coincide.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            store_policy_state (bool, True): whether the policy state is stored;
            device (str, None): torch device used by the buffer.

        Returns:
            The dataset info.

        """
        backend = agent_info.backend
        array_backend = ArrayBackend.get_array_backend(backend)
        device = device if backend == 'torch' else None
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = array_backend.to_backend_dtype(mdp_info.observation_space.data_type)
        action_shape = mdp_info.action_space.shape
        action_dtype = array_backend.to_backend_dtype(mdp_info.action_space.data_type)
        policy_state_shape = agent_info.policy_state_shape if store_policy_state else None

        return DatasetInfo(backend, backend, device, device, horizon, gamma, state_shape, state_dtype,
                           action_shape, action_dtype, policy_state_shape)


class Dataset(MushroomObject):
    """
    Collection of the transitions gathered while an agent interacts with an environment. The data is split into
    two backend-aware groups, each delegated to a backend-specific columnar container (``NumpyDataset``,
    ``TorchDataset`` or ``ListDataset``):

    - The environment data (state, action, reward, next state, absorbing and last flags), kept in the
      environment backend;
    - The agent data (policy state and next policy state), kept in the agent backend so that it
      never needs a per-step conversion. Not created when the agent is stateless.

    Step info, episode info and the per-episode policy parameters (``theta``) are stored alongside
    the transitions.

    """
    class _Field(IntEnum):
        STATE = 0
        ACTION = 1
        REWARD = 2
        NEXT_STATE = 3
        ABSORBING = 4
        LAST = 5

    class _PolicyField(IntEnum):
        POLICY_STATE = 0
        POLICY_NEXT_STATE = 1

    def __init__(self, dataset_info, n_steps=None, n_episodes=None, core_counts_episodes=False):
        """
        Constructor. Exactly one of ``n_steps`` and ``n_episodes`` must be given; it sizes the preallocated
        containers (the list backend grows on demand instead).

        Args:
            dataset_info (DatasetInfo): the static information used to build the dataset;
            n_steps (int, None): number of steps the dataset is allocated for;
            n_episodes (int, None): number of episodes the dataset is allocated for;
            core_counts_episodes (bool, False): whether the collecting core counts episodes, which needs a
                slightly larger preallocation.

        """
        assert (n_steps is not None and n_episodes is None) or (n_steps is None and n_episodes is not None)

        self._dataset_info = dataset_info

        info_n_envs = min(n_episodes, dataset_info.n_envs) if n_episodes else dataset_info.n_envs
        vectorized = dataset_info.n_envs > 1
        self._info = ExtraInfo(info_n_envs, dataset_info.env_backend, dataset_info.env_device, vectorized=vectorized)
        self._episode_info = ExtraInfo(info_n_envs, dataset_info.env_backend, dataset_info.env_device,
                                       vectorized=vectorized)
        self._theta_list = list()

        n_envs = (min(n_episodes, dataset_info.n_envs) if n_episodes else dataset_info.n_envs) if vectorized else None

        self._base_shape = self._compute_base_shape(dataset_info, n_steps, n_episodes, core_counts_episodes)

        env_shapes, env_dtypes = self._env_specs(dataset_info, self._base_shape)
        self._data = self._make_container(dataset_info.env_backend, env_shapes, env_dtypes,
                                          dataset_info.env_device, n_envs)

        if dataset_info.policy_state_shape is not None:
            policy_shapes, policy_dtypes = self._policy_specs(dataset_info, self._base_shape)
            # the policy state lives in the agent backend, but a list (infinite-horizon) env keeps it growable
            agent_backend = 'list' if dataset_info.env_backend == 'list' else dataset_info.agent_backend
            self._agent_data = self._make_container(agent_backend, policy_shapes, policy_dtypes,
                                                    dataset_info.agent_device, n_envs)
        else:
            self._agent_data = None

        self._add_all_save_attr()

    @staticmethod
    def _compute_base_shape(dataset_info, n_steps, n_episodes, core_counts_episodes):
        if dataset_info.env_backend == 'list':
            return None

        if n_steps is not None:
            n_samples = n_steps
        else:
            horizon = dataset_info.horizon
            assert np.isfinite(horizon)
            n_samples = horizon * n_episodes

        if dataset_info.n_envs == 1:
            return (n_samples,)
        elif n_episodes:
            horizon = dataset_info.horizon
            x = math.ceil(n_episodes / dataset_info.n_envs)
            return (x * horizon, min(n_episodes, dataset_info.n_envs))
        elif core_counts_episodes:
            return (math.ceil(n_samples / dataset_info.n_envs) + 1 + dataset_info.horizon, dataset_info.n_envs)
        else:
            return (math.ceil(n_samples / dataset_info.n_envs) + 1, dataset_info.n_envs)

    @staticmethod
    def _env_specs(dataset_info, base_shape):
        backend = dataset_info.env_array_backend
        base = base_shape if base_shape is not None else ()
        state_shape = base + dataset_info.state_shape
        action_shape = base + dataset_info.action_shape

        shapes = [state_shape, action_shape, base, state_shape, base, base]
        dtypes = [backend.to_backend_dtype(dataset_info.state_dtype),
                  backend.to_backend_dtype(dataset_info.action_dtype),
                  backend.to_backend_dtype(float),
                  backend.to_backend_dtype(dataset_info.state_dtype),
                  backend.to_backend_dtype(bool),
                  backend.to_backend_dtype(bool)]
        return shapes, dtypes

    @staticmethod
    def _policy_specs(dataset_info, base_shape):
        backend = dataset_info.agent_array_backend
        base = base_shape if base_shape is not None else ()
        policy_shape = base + dataset_info.policy_state_shape

        shapes = [policy_shape, policy_shape]
        dtypes = [backend.to_backend_dtype(float), backend.to_backend_dtype(float)]
        return shapes, dtypes

    @staticmethod
    def _make_container(backend_name, shapes, dtypes, device=None, n_envs=None):
        if backend_name == 'numpy':
            return NumpyDataset(shapes, dtypes, n_envs=n_envs)
        elif backend_name == 'torch':
            return TorchDataset(shapes, dtypes, device=device, n_envs=n_envs)
        else:
            return ListDataset(len(shapes), n_envs=n_envs)

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

        new_dataset._dataset_info = dataset._dataset_info if dataset is not None else None

        new_dataset._base_shape = None
        new_dataset._info = None
        new_dataset._episode_info = None
        new_dataset._data = None
        new_dataset._agent_data = None
        new_dataset._theta_list = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_state=None, policy_next_state=None, info=None, episode_info=None, theta_list=None,
                   horizon=None, gamma=0.99, backend='numpy', policy_backend=None, device=None):
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
            backend (str, 'numpy'): backend to be used by the dataset;
            policy_backend (str, None): backend to be used for the policy state arrays; defaults to ``backend``.

        Returns:
            The list of transitions.

        """
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(absorbings) == len(lasts)

        if policy_state is not None:
            assert len(states) == len(policy_state) == len(policy_next_state)

        if policy_backend is None:
            policy_backend = backend

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

        env_class = cls._container_class(backend)
        dataset._data = env_class.from_array([states, actions, rewards, next_states, absorbings, lasts])

        if policy_state is not None:
            policy_class = cls._container_class(policy_backend)
            dataset._agent_data = policy_class.from_array([policy_state, policy_next_state])
        else:
            dataset._agent_data = None

        state_shape = cls._infer_shape(states)
        action_shape = cls._infer_shape(actions)
        state_dtype = cls._infer_dtype(states)
        action_dtype = cls._infer_dtype(actions)
        policy_state_shape = None if policy_state is None else cls._infer_shape(policy_state)

        dataset._dataset_info = DatasetInfo(backend, policy_backend, device, None, horizon, gamma,
                                            state_shape, state_dtype, action_shape, action_dtype, policy_state_shape)

        return dataset

    @staticmethod
    def _container_class(backend_name):
        if backend_name == 'numpy':
            return NumpyDataset
        elif backend_name == 'torch':
            return TorchDataset
        else:
            return ListDataset

    def _store_step(self, step):
        self._data.append(*step[:len(self._Field)])
        if self._agent_data is not None:
            self._agent_data.append(*step[len(self._Field):])

    def append(self, step, info):
        self._store_step(step)
        self._info.append(info)

    def append_batch(self, other):
        """
        Append all transitions from another dataset without copying data.

        Args:
            other (Dataset): dataset whose transitions will be appended.

        """
        self._data.append_batch(other._data)
        if self._agent_data is not None:
            self._agent_data.append_batch(other._agent_data)
        self._info += other._info

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
        if self._agent_data is not None:
            self._agent_data.clear()

    def get_view(self, index, copy=False):
        dataset = self.create_raw_instance(dataset=self)

        dataset._info = self._info.get_view(index, copy)
        dataset._episode_info = self._episode_info.get_view(index, copy)
        dataset._data = self._data.get_view(index, copy)
        dataset._agent_data = self._agent_data.get_view(index, copy) if self._agent_data is not None else None
        dataset._theta_list = []

        return dataset

    def item(self):
        assert len(self) == 1
        return self[0]

    def __getitem__(self, index):
        if isinstance(index, (slice, np.ndarray, torch.Tensor)):
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
        result._agent_data = (self._agent_data + other._agent_data) if self._agent_data is not None else None

        result._data.column(self._Field.LAST)[len(self) - 1] = True

        return result

    def __len__(self):
        return len(self._data)

    @property
    def state(self):
        return self._data.column(self._Field.STATE)

    @property
    def action(self):
        return self._data.column(self._Field.ACTION)

    @property
    def reward(self):
        return self._data.column(self._Field.REWARD)

    @property
    def next_state(self):
        return self._data.column(self._Field.NEXT_STATE)

    @property
    def absorbing(self):
        return self._data.column(self._Field.ABSORBING)

    @property
    def last(self):
        return self._data.column(self._Field.LAST)

    @property
    def policy_state(self):
        return self._agent_data.column(self._PolicyField.POLICY_STATE)

    @property
    def policy_next_state(self):
        return self._agent_data.column(self._PolicyField.POLICY_NEXT_STATE)

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
            An array with the length of each episode in the dataset.

        """
        lengths = list()
        length = 0
        for sample in self:
            length += 1
            if sample[-1] == 1:
                lengths.append(length)
                length = 0

        return self._dataset_info.env_array_backend.as_array(lengths)

    @property
    def n_episodes(self):
        return self._data.n_episodes(self._Field.LAST)

    @property
    def undiscounted_return(self):
        return self.compute_J()

    @property
    def discounted_return(self):
        return self.compute_J(self._dataset_info.gamma)

    @property
    def array_backend(self):
        return self._dataset_info.env_array_backend

    @property
    def is_stateful(self):
        return self._agent_data is not None

    def parse(self, to=None):
        """
        Return the dataset as set of arrays.
        Args:
            to (str, None):  the backend to be used for the returned arrays. By default, the dataset backend is used.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, next state, absorbing and last

        """
        if to is None:
            to = self._dataset_info.env_array_backend.get_backend_name()
        return self._convert(self.state, self.action, self.reward, self.next_state, self.absorbing, self.last, to=to)

    def parse_policy_state(self, to=None):
        """
        Return the policy state arrays of the dataset.

        Args:
            to (str, None): the backend to be used for the returned arrays. By default, the policy's backend is used.

        Returns:
            A tuple containing the policy state and policy next state arrays.

        """
        backend = self._dataset_info.agent_array_backend
        if to is None:
            to = backend.get_backend_name()
        return self._convert(self.policy_state, self.policy_next_state, to=to, backend=backend)

    def to_backend(self, backend):
        """
        Return a copy of this dataset converted to the given backend.

        Args:
            backend (str): target backend (``'numpy'``, ``'torch'``, or ``'list'``).

        Returns:
            A new Dataset in the requested backend, or ``self`` if the backend already matches.

        """
        if self._dataset_info.env_array_backend.get_backend_name() == backend \
                and self._dataset_info.agent_array_backend.get_backend_name() == backend:
            return self
        state, action, reward, next_state, absorbing, last = self.parse(to=backend)
        policy_state, policy_next_state = (self.parse_policy_state(to=backend) if self.is_stateful else (None, None))
        return Dataset.from_array(state, action, reward, next_state, absorbing, last,
                                  policy_state=policy_state, policy_next_state=policy_next_state,
                                  info=self._info, episode_info=self._episode_info,
                                  theta_list=self._theta_list, backend=backend, policy_backend=backend)

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
        return self._dataset_info.env_array_backend.from_list(x_0)

    def compute_J(self, gamma=1.):
        """
        Compute the cumulative discounted reward of each episode in the dataset.

        Args:
            gamma (float, 1.): discount factor.

        Returns:
            The cumulative discounted reward of each episode in the dataset.

        """
        backend = self._dataset_info.env_array_backend
        r_ep = split_episodes(backend.as_array(self.last), backend.as_array(self.reward))

        if len(r_ep.shape) == 1:
            r_ep = backend.expand_dims(r_ep, 0)
        if self._dataset_info.env_backend == 'torch':
            js = backend.zeros(r_ep.shape[0], dtype=r_ep.dtype, device=r_ep.device)
        else:
            js = backend.zeros(r_ep.shape[0], dtype=r_ep.dtype)

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
            median = self._dataset_info.env_array_backend.median(J)
            return J.min(), J.max(), J.mean(), median, len(J)
        else:
            return 0, 0, 0, 0, 0

    def _convert(self, *arrays, to='numpy', backend=None):
        backend = backend if backend is not None else self._dataset_info.env_array_backend
        if to == 'numpy':
            return backend.arrays_to_numpy(*arrays)
        elif to == 'torch':
            return backend.arrays_to_torch(*arrays)
        elif to == 'list':
            return backend.arrays_to_list(*arrays)
        else:
            raise NotImplementedError

    def _add_all_save_attr(self):
        self._add_save_attr(
            _info='mushroom',
            _episode_info='mushroom',
            _theta_list='pickle',
            _data='mushroom',
            _agent_data='mushroom',
            _base_shape='primitive',
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

    @staticmethod
    def _infer_shape(data):
        if hasattr(data, 'shape'):
            return data.shape[1:]
        return data[0].shape if len(data) and hasattr(data[0], 'shape') else ()

    @staticmethod
    def _infer_dtype(data):
        if hasattr(data, 'dtype'):
            return data.dtype
        return data[0].dtype if len(data) and hasattr(data[0], 'dtype') else None


class VectorizedDataset(Dataset):
    """
    :class:`Dataset` variant for data collected from several environments in parallel. Each step stores a batch
    of transitions together with a boolean ``mask`` (kept in its own env-backend container) marking which
    environments were active. The padded per-environment episodes are turned back into a flat :class:`Dataset`
    with :meth:`flatten`, typically once per fit.

    """
    def __init__(self, dataset_info, n_steps=None, n_episodes=None, core_counts_episodes=False):
        super().__init__(dataset_info, n_steps, n_episodes, core_counts_episodes)

        mask_shape = self._base_shape if self._base_shape is not None else ()
        self._mask_data = self._make_container(dataset_info.env_backend, [mask_shape],
                                               [self._dataset_info.env_array_backend.to_backend_dtype(bool)],
                                               dataset_info.env_device, self._data.n_envs)

        self._initialize_theta_list(self._dataset_info.n_envs)

    def append(self, step, info):
        raise RuntimeError("Trying to use append on a vectorized dataset")

    def append_vectorized(self, step, info, mask):
        """
        Append one step of a batch of parallel environments.

        Args:
            step (tuple): the batched transition, one entry per environment field (plus the policy state fields
                when the agent is stateful);
            info (dict): the batched step info;
            mask (Array): boolean mask selecting the environments that are currently active.

        """
        self._store_step(step)
        self._mask_data.append(mask)
        self._info.append(info)

    def append_theta_vectorized(self, theta, mask):
        """
        Append the policy parameters of the active environments to their per-environment ``theta`` lists.

        Args:
            theta (Array): the policy parameters, one entry per environment;
            mask (Array): boolean mask selecting the environments that are currently active.

        """
        for i in range(len(theta)):
            if mask[i]:
                self._theta_list[i].append(theta[i])

    def clear(self, n_steps_per_fit=None):
        """
        Clear the dataset. When ``n_steps_per_fit`` is given and more than that many (masked) steps were
        collected, the surplus tail is carried forward into the freshly cleared dataset so the next fit starts
        with it.

        Args:
            n_steps_per_fit (int, None): number of steps consumed by the fit; the rest is carried forward.

        Returns:
            The number of steps carried forward.

        """
        n_envs = len(self._theta_list)
        n_carry_forward_steps = 0

        residual_data = None
        residual_agent_data = None
        residual_mask_data = None
        if n_steps_per_fit is not None:
            n_steps_dataset = self.mask.sum().item()

            if n_steps_dataset > n_steps_per_fit:
                n_extra_steps = n_steps_dataset - n_steps_per_fit
                n_parallel_steps = int(np.ceil(n_extra_steps / self._dataset_info.n_envs))
                view_size = slice(-n_parallel_steps, None)

                residual_data = self._data.get_view(view_size, copy=True)
                if self._agent_data is not None:
                    residual_agent_data = self._agent_data.get_view(view_size, copy=True)
                residual_mask_data = self._mask_data.get_view(view_size, copy=True)

                mask = self._dataset_info.env_array_backend.as_array(residual_mask_data.column())
                original_shape = mask.shape
                mask = mask.flatten()
                true_indices = self._dataset_info.env_array_backend.where(mask)[0]
                mask[true_indices[n_extra_steps:]] = False

                mask_column = residual_mask_data.column()
                new_mask = mask.reshape(original_shape)
                if isinstance(mask_column, list):
                    mask_column[:] = list(new_mask)
                else:
                    mask_column[:] = new_mask

                residual_info = self._info.get_view(view_size, copy=True)
                residual_episode_info = self._episode_info.get_view(view_size, copy=True)

                n_carry_forward_steps = mask.sum()

        super().clear()
        self._mask_data.clear()
        self._initialize_theta_list(n_envs)

        if n_steps_per_fit is not None and residual_data is not None:
            self._data = residual_data
            self._agent_data = residual_agent_data
            self._mask_data = residual_mask_data
            self._info = residual_info
            self._episode_info = residual_episode_info

        return n_carry_forward_steps

    def flatten(self, n_steps_per_fit=None):
        """
        Turn the padded per-environment data into a flat :class:`Dataset`, dropping the inactive entries via the
        mask and concatenating the environments end to end.

        Args:
            n_steps_per_fit (int, None): if given, keep only the first this many flattened steps.

        Returns:
            A flat :class:`Dataset`, or ``None`` if the dataset is empty.

        """
        if len(self) == 0:
            return None

        mask = self.mask
        env_backend = self._dataset_info.env_array_backend
        agent_backend = self._dataset_info.agent_array_backend

        states = env_backend.pack_padded_sequence(self.state, mask)
        actions = env_backend.pack_padded_sequence(self.action, mask)
        rewards = env_backend.pack_padded_sequence(self.reward, mask)
        next_states = env_backend.pack_padded_sequence(self.next_state, mask)
        absorbings = env_backend.pack_padded_sequence(self.absorbing, mask)

        last_padded = env_backend.as_array(self.last)
        last_padded[-1, :] = True
        lasts = env_backend.pack_padded_sequence(last_padded, mask)

        policy_state = None
        policy_next_state = None

        if self.is_stateful:
            policy_mask = agent_backend.convert_to_backend(env_backend, mask)
            policy_state = agent_backend.pack_padded_sequence(self.policy_state, policy_mask)
            policy_next_state = agent_backend.pack_padded_sequence(self.policy_next_state, policy_mask)

        if n_steps_per_fit is not None:
            states = states[:n_steps_per_fit]
            actions = actions[:n_steps_per_fit]
            rewards = rewards[:n_steps_per_fit]
            next_states = next_states[:n_steps_per_fit]
            absorbings = absorbings[:n_steps_per_fit]
            lasts = lasts[:n_steps_per_fit]

            if self.is_stateful:
                policy_state = policy_state[:n_steps_per_fit]
                policy_next_state = policy_next_state[:n_steps_per_fit]

        flat_theta_list = self._flatten_theta_list()

        flat_info = self._info.flatten(mask)
        flat_episode_info = self._episode_info.flatten(mask)

        return Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                  policy_state=policy_state, policy_next_state=policy_next_state,
                                  info=flat_info, episode_info=flat_episode_info, theta_list=flat_theta_list,
                                  horizon=self._dataset_info.horizon, gamma=self._dataset_info.gamma,
                                  backend=env_backend.get_backend_name(),
                                  policy_backend=agent_backend.get_backend_name())

    def get_view(self, index, copy=False):
        dataset = super().get_view(index, copy)
        dataset._mask_data = self._mask_data.get_view(index, copy)
        return dataset

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
        """
        Boolean mask marking, for every stored step, which environments were active.

        """
        return self._dataset_info.env_array_backend.as_array(self._mask_data.column())

    def _add_all_save_attr(self):
        super()._add_all_save_attr()
        self._add_save_attr(
            _mask_data='mushroom'
        )
