import numpy as np
import math

from enum import IntEnum

import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.extra_info import ExtraInfo

from ._impl import NumpyDataset, TorchDataset, ListDataset

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

    def flat(self):
        """
        Returns:
            A copy of this dataset info describing a single, non-vectorized environment.

        """
        return DatasetInfo(self.env_backend, self.agent_backend, self.env_device, self.agent_device, self.horizon,
                           self.gamma, self.state_shape, self.state_dtype, self.action_shape, self.action_dtype,
                           self.policy_state_shape)

    @staticmethod
    def create_dataset_info(mdp_info, agent_info, n_envs=1):
        """
        Build the dataset info for on-policy collection: the environment data uses ``mdp_info.backend`` (forced
        to ``'list'`` for infinite-horizon MDPs) and the agent data uses ``agent_info.backend``.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            n_envs (int, 1): number of parallel environments.

        Returns:
            The dataset info.

        """
        env_backend = mdp_info.backend
        if not np.isfinite(mdp_info.horizon):
            assert env_backend != 'torch', "Infinite-horizon collection is not supported for the torch backend."
            assert agent_info.policy_state_shape is None or agent_info.backend != 'torch', \
                "Infinite-horizon collection is not supported for a stateful torch agent."
            env_backend = 'list'
        env_device = mdp_info.device
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = mdp_info.observation_space.data_type
        action_shape = mdp_info.action_space.shape
        action_dtype = mdp_info.action_space.data_type
        policy_state_shape = agent_info.policy_state_shape
        agent_device = agent_info.device

        return DatasetInfo(env_backend, agent_info.backend, env_device, agent_device, horizon, gamma,
                           state_shape, state_dtype, action_shape, action_dtype, policy_state_shape, n_envs=n_envs)

    @staticmethod
    def create_replay_memory_info(mdp_info, agent_info, store_policy_state=True):
        """
        Build the dataset info for a replay memory: the whole buffer (both the transition data and the policy
        state) lives in the agent backend, so the environment and agent backends/devices coincide.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            store_policy_state (bool, True): whether the policy state is stored.

        Returns:
            The dataset info.

        """
        backend = agent_info.backend
        array_backend = ArrayBackend.get_array_backend(backend)
        device = agent_info.device
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = array_backend.to_backend_dtype(mdp_info.observation_space.data_type)
        action_shape = mdp_info.action_space.shape
        action_dtype = array_backend.to_backend_dtype(mdp_info.action_space.data_type)
        policy_state_shape = agent_info.policy_state_shape if store_policy_state else None

        return DatasetInfo(backend, backend, device, device, horizon, gamma, state_shape, state_dtype,
                           action_shape, action_dtype, policy_state_shape)

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
        self._extras = ExtraInfo(info_n_envs, dataset_info.env_backend, dataset_info.env_device,
                                 vectorized=vectorized, theta_backend=dataset_info.agent_backend,
                                 theta_device=dataset_info.agent_device)

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

    def __getitem__(self, index):
        if isinstance(index, (slice, np.ndarray, torch.Tensor)):
            return self.get_view(index)
        elif isinstance(index, int) and index < len(self._data):
            return self._data[index]
        else:
            raise IndexError

    def __add__(self, other):
        result = self.create_raw_instance(dataset=self)

        result._extras = self._extras + other._extras
        result._data = self._data + other._data
        result._agent_data = (self._agent_data + other._agent_data) if self._agent_data is not None else None

        return result

    def __iadd__(self, other):
        capacity = self._data.capacity
        if capacity is not None and len(self) + len(other) > capacity:
            return self + other

        self.append_batch(other)
        self._extras += other._extras

        return self

    def __len__(self):
        return len(self._data)

    def append(self, step, info):
        self._store_step(step)
        self._extras.append_step(info)

    def append_batch(self, other):
        """
        Append the transitions of another dataset in place without copying data. Only the transition data is
        merged (env data and policy state); the step information, the per-episode information and the policy
        parameters are not updated.

        Args:
            other (Dataset): dataset whose transitions will be appended.

        """
        self._data.append_batch(other._data)
        if self._agent_data is not None:
            self._agent_data.append_batch(other._agent_data)

    def reserve(self, capacity):
        """
        Ensure the dataset can hold at least ``capacity`` transitions, reallocating a larger buffer and copying
        the stored transitions across when needed. It is a no-op when the buffer is already large enough or
        unbounded (list backend).

        Args:
            capacity (int): the minimum number of transitions the dataset must be able to hold.

        """
        self._data.reserve(capacity)
        if self._agent_data is not None:
            self._agent_data.reserve(capacity)

    def append_episode_info(self, info, mask=None):
        """
        Append the information an environment reported when resetting.

        Args:
            info (dict or list): the information the reset reported;
            mask (Array, None): boolean mask selecting the environments that were reset.

        """
        self._extras.append_episode(info, mask)

    def append_theta(self, theta):
        """
        Append the policy parameters of the episode that is starting.

        Args:
            theta (Array): the policy parameters.

        """
        self._extras.append_theta(theta)

    def get_info(self, field, index=None):
        if index is None:
            return self.info[field]
        else:
            return self.info[field][index]

    def clear(self):
        self._extras.clear()

        self._data.clear()
        if self._agent_data is not None:
            self._agent_data.clear()

    def get_view(self, index, copy=False):
        dataset = self.create_raw_instance(dataset=self)

        dataset._extras = self._extras.get_view(index, copy)
        dataset._data = self._data.get_view(index, copy)
        dataset._agent_data = self._agent_data.get_view(index, copy) if self._agent_data is not None else None

        return dataset

    def item(self):
        assert len(self) == 1
        return self[0]

    def parse(self, to=None, device=None):
        """
        Return the dataset as set of arrays.
        Args:
            to (str, None):  the backend to be used for the returned arrays. By default, the dataset backend is used;
            device (str, None): device the returned arrays are placed on, or ``None`` for the default one.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, next state, absorbing and last

        """
        if to is None:
            to = self._dataset_info.env_array_backend.get_backend_name()
        return self._convert(self.state, self.action, self.reward, self.next_state, self.absorbing, self.last,
                             to=to, device=device)

    def parse_policy_state(self, to=None, device=None):
        """
        Return the policy state arrays of the dataset.

        Args:
            to (str, None): the backend to be used for the returned arrays. By default, the policy's backend is used;
            device (str, None): device the returned arrays are placed on, or ``None`` for the default one.

        Returns:
            A tuple containing the policy state and policy next state arrays.

        """
        backend = self._dataset_info.agent_array_backend
        if to is None:
            to = backend.get_backend_name()
        return self._convert(self.policy_state, self.policy_next_state, to=to, backend=backend, device=device)

    def to_backend(self, backend, device=None):
        """
        Return a copy of this dataset converted to the given backend.

        Args:
            backend (str): target backend (``'numpy'``, ``'torch'``, or ``'list'``);
            device (str, None): device the converted dataset is placed on, or ``None`` for the default one.

        Returns:
            A new Dataset in the requested backend, or ``self`` if the backend and the device already match.

        """
        if self._dataset_info.env_array_backend.get_backend_name() == backend \
                and self._dataset_info.agent_array_backend.get_backend_name() == backend \
                and device in (None, self._dataset_info.env_device) \
                and device in (None, self._dataset_info.agent_device):
            return self
        state, action, reward, next_state, absorbing, last = self.parse(to=backend, device=device)
        policy_state, policy_next_state = (self.parse_policy_state(to=backend, device=device) if self.is_stateful
                                           else (None, None))
        return Dataset.from_array(state, action, reward, next_state, absorbing, last,
                                  policy_state=policy_state, policy_next_state=policy_next_state,
                                  extras=self._extras.to_backend(backend, device), backend=backend,
                                  policy_backend=backend, device=device, agent_device=device)

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
        return self._dataset_info.env_array_backend.from_list(x_0, device=self._dataset_info.env_device)

    def compute_J(self, gamma=1., skip_incomplete=True):
        """
        Compute the cumulative discounted reward of each episode in the dataset.

        Args:
            gamma (float, 1.): discount factor;
            skip_incomplete (bool, True): whether to leave out the trailing episode when the dataset ends before
                it does.

        Returns:
            The cumulative discounted reward of each episode in the dataset.

        """
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device

        last = backend.as_array(self.last, device=device)
        reward = backend.as_array(self.reward, device=device)

        if skip_incomplete:
            last_idx = backend.nonzero(last)
            n_complete_steps = last_idx[-1].item() + 1 if len(last_idx) > 0 else 0
            last = last[:n_complete_steps]
            reward = reward[:n_complete_steps]

        if len(last) == 0:
            return backend.zeros(0, device=device)

        _, r_ep = split_episodes(last, reward)

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
            A dictionary with the number of completed episodes under ``n_episodes`` and, unless that number is
            zero, the minimum, maximum, mean and median return reached in an episode under ``min_J``, ``max_J``,
            ``mean_J`` and ``median_J``.

        """
        J = self.compute_J(gamma)

        if len(J) > 0:
            median = self._dataset_info.env_array_backend.median(J)
            return dict(min_J=J.min(), max_J=J.max(), mean_J=J.mean(), median_J=median, n_episodes=len(J))
        else:
            return dict(n_episodes=0)

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
        new_dataset._extras = None
        new_dataset._data = None
        new_dataset._agent_data = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_state=None, policy_next_state=None, extras=None,
                   horizon=None, gamma=0.99, backend='numpy', policy_backend=None, device=None, agent_device=None):
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
            extras (ExtraInfo, None): step info, episode info and policy parameters to copy into the dataset;
            horizon (int, None): horizon of the mdp;
            gamma (float, 0.99): discount factor;
            backend (str, 'numpy'): backend to be used by the dataset;
            policy_backend (str, None): backend to be used for the policy state arrays; defaults to ``backend``;
            device (str, None): device the environment arrays are stored on, or ``None`` for the default one;
            agent_device (str, None): device the policy state arrays are stored on, or ``None`` for the default one.

        Returns:
            The list of transitions.

        """
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(absorbings) == len(lasts)

        if policy_state is not None:
            assert len(states) == len(policy_state) == len(policy_next_state)

        if policy_backend is None:
            policy_backend = backend

        dataset = cls.create_raw_instance()

        dataset._extras = ExtraInfo(1, backend) if extras is None else extras.copy()

        env_arrays = [states, actions, rewards, next_states, absorbings, lasts]
        dataset._data = cls._container_from_array(backend, env_arrays, device)

        if policy_state is not None:
            agent_arrays = [policy_state, policy_next_state]
            dataset._agent_data = cls._container_from_array(policy_backend, agent_arrays, agent_device)
        else:
            dataset._agent_data = None

        state_shape = cls._infer_shape(states)
        action_shape = cls._infer_shape(actions)
        state_dtype = cls._infer_dtype(states)
        action_dtype = cls._infer_dtype(actions)
        policy_state_shape = None if policy_state is None else cls._infer_shape(policy_state)

        dataset._dataset_info = DatasetInfo(backend, policy_backend, device, agent_device, horizon, gamma,
                                            state_shape, state_dtype, action_shape, action_dtype, policy_state_shape)

        return dataset

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
        """
        Returns:
            A flat dictionary holding an array per key of the step information. It describes the dataset as of
            this call and does not update as further steps are collected.

        """
        return self._extras.parse_steps()

    @property
    def episode_info(self):
        """
        Returns:
            A flat dictionary holding an array per key of the episode information. It describes the dataset as
            of this call and does not update as further episodes are collected.

        """
        return self._extras.parse_episodes()

    @property
    def theta_list(self):
        """
        Returns:
            The policy parameters, one entry per episode, kept in one list per environment when they are
            collected from a vectorized environment.

        """
        return self._extras.theta

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

        return self._dataset_info.env_array_backend.as_array(lengths, device=self._dataset_info.env_device)

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

    @property
    def capacity(self):
        """
        The number of transitions the dataset can hold, or ``None`` when it grows without bound (list backend).

        """
        return self._data.capacity

    def _store_step(self, step):
        self._data.append(*step[:len(self._Field)])
        if self._agent_data is not None:
            self._agent_data.append(*step[len(self._Field):])

    def _convert(self, *arrays, to='numpy', backend=None, device=None):
        backend = backend if backend is not None else self._dataset_info.env_array_backend
        if to == 'numpy':
            ArrayBackend.get_array_backend(to).check_device(device)
            return backend.arrays_to_numpy(*arrays)
        elif to == 'torch':
            return backend.arrays_to_torch(*arrays, device=device)
        elif to == 'list':
            ArrayBackend.get_array_backend(to).check_device(device)
            return backend.arrays_to_list(*arrays)
        else:
            raise NotImplementedError

    def _add_all_save_attr(self):
        self._add_save_attr(
            _extras='mushroom',
            _data='mushroom',
            _agent_data='mushroom',
            _base_shape='primitive',
            _dataset_info='mushroom'
        )

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
            return x * horizon, min(n_episodes, dataset_info.n_envs)
        elif core_counts_episodes:
            return math.ceil(n_samples / dataset_info.n_envs) + 1 + dataset_info.horizon, dataset_info.n_envs
        else:
            return math.ceil(n_samples / dataset_info.n_envs) + 1, dataset_info.n_envs

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

    @staticmethod
    def _container_from_array(backend_name, arrays, device=None):
        if backend_name == 'numpy':
            return NumpyDataset.from_array(arrays)
        elif backend_name == 'torch':
            return TorchDataset.from_array(arrays, device=device)
        else:
            return ListDataset.from_array(arrays)

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

    def __add__(self, other):
        result = super().__add__(other)
        result._mask_data = self._mask_data + other._mask_data

        return result

    def append(self, step, info):
        raise RuntimeError("Trying to use append on a vectorized dataset")

    def append_batch(self, other):
        super().append_batch(other)
        self._mask_data.append_batch(other._mask_data)

    def reserve(self, capacity):
        super().reserve(capacity)
        self._mask_data.reserve(capacity)

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
        self._extras.append_step(info)

    def append_theta_vectorized(self, theta, mask):
        """
        Append the policy parameters of the active environments to their per-environment ``theta`` lists.

        Args:
            theta (Array): the policy parameters, one entry per environment;
            mask (Array): boolean mask selecting the environments that are currently active.

        """
        self._extras.append_theta_vectorized(theta, mask)

    def consume(self, n_steps):
        """
        Split off the first ``n_steps`` steps in collection (row-major) order (or all of them when ``n_steps``
        is ``None``): return a lightweight vectorized dataset that shares this one's data but whose mask keeps
        only those consumed steps, and mark those same steps inactive in this dataset's own mask so that only the
        leftover remains active.

        Args:
            n_steps (int, None): number of steps to consume.

        Returns:
            A vectorized dataset masked to the consumed steps.

        """
        backend = self._dataset_info.env_array_backend
        mask = self.mask
        active = backend.where(mask.reshape(-1))[0]
        n_steps = len(active) if n_steps is None else n_steps
        assert 0 <= n_steps <= len(active)

        consumed_mask = backend.copy(mask)
        consumed_mask.reshape(-1)[active[n_steps:]] = False

        leftover_mask = backend.copy(mask)
        leftover_mask.reshape(-1)[active[:n_steps]] = False
        mask_column = self._mask_data.column()
        if isinstance(mask_column, list):
            mask_column[:] = list(leftover_mask)
        else:
            mask_column[:] = leftover_mask

        view = self.create_raw_instance(dataset=self)
        view._extras = self._extras
        view._data = self._data
        view._agent_data = self._agent_data
        view._mask_data = self._mask_data.from_array([consumed_mask])

        return view

    def clear(self, keep_leftovers=False):
        """
        Clear the dataset. By default, the whole dataset is wiped. With ``keep_leftovers=True`` the steps still
        active after a :meth:`consume` (the leftover the fit did not consume) are compacted to the front and
        kept, so the next fit starts with them.

        Args:
            keep_leftovers (bool, False): whether to keep the leftover steps instead of wiping everything.

        Returns:
            The number of steps kept.

        """
        if keep_leftovers:
            backend = self._dataset_info.env_array_backend
            row_active = backend.sum(self.mask, dim=1)
            n_carry = int(row_active.sum().item())

            if n_carry > 0:
                split_row = len(row_active) - int((row_active > 0).sum().item())
                self._data.compact(split_row)
                if self._agent_data is not None:
                    self._agent_data.compact(split_row)
                self._mask_data.compact(split_row)
                self._extras.keep_from(split_row)

                return n_carry

        super().clear()
        self._mask_data.clear()

        return 0

    def flatten(self):
        """
        Turn the padded per-environment data into a flat :class:`Dataset`, dropping the inactive entries via the
        mask and concatenating the environments end to end.

        Returns:
            A flat :class:`Dataset`.

        """
        if len(self) == 0:
            return Dataset(self._dataset_info.flat(), n_steps=0)

        mask = self.mask
        env_backend = self._dataset_info.env_array_backend
        agent_backend = self._dataset_info.agent_array_backend

        states = env_backend.pack_padded_sequence(self.state, mask)
        actions = env_backend.pack_padded_sequence(self.action, mask)
        rewards = env_backend.pack_padded_sequence(self.reward, mask)
        next_states = env_backend.pack_padded_sequence(self.next_state, mask)
        absorbings = env_backend.pack_padded_sequence(self.absorbing, mask)

        steps = env_backend.expand_dims(env_backend.arange(0, len(mask), device=self._dataset_info.env_device), 1)
        block_end = mask & (steps == env_backend.max(env_backend.where(mask, steps, -1), dim=0))

        last_padded = env_backend.copy(env_backend.as_array(self.last, device=self._dataset_info.env_device))
        last_padded[block_end] = True
        lasts = env_backend.pack_padded_sequence(last_padded, mask)

        policy_state = None
        policy_next_state = None

        if self.is_stateful:
            policy_mask = agent_backend.convert_mask(mask, backend=env_backend, device=self._dataset_info.agent_device)
            policy_state = agent_backend.pack_padded_sequence(self.policy_state, policy_mask)
            policy_next_state = agent_backend.pack_padded_sequence(self.policy_next_state, policy_mask)

        flat_extras = self._extras.flatten(mask)

        return Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                  policy_state=policy_state, policy_next_state=policy_next_state,
                                  extras=flat_extras, horizon=self._dataset_info.horizon,
                                  gamma=self._dataset_info.gamma,
                                  backend=env_backend.get_backend_name(),
                                  policy_backend=agent_backend.get_backend_name(),
                                  device=self._dataset_info.env_device,
                                  agent_device=self._dataset_info.agent_device)

    def get_view(self, index, copy=False):
        dataset = super().get_view(index, copy)
        dataset._mask_data = self._mask_data.get_view(index, copy)
        return dataset

    @property
    def capacity(self):
        capacity = super().capacity
        if capacity is None:
            return None
        return min(capacity, self._mask_data.capacity)

    @property
    def mask(self):
        """
        Boolean mask marking, for every stored step, which environments were active.

        """
        return self._dataset_info.env_array_backend.convert_mask(self._mask_data.column(),
                                                                 device=self._dataset_info.env_device)

    def _add_all_save_attr(self):
        super()._add_all_save_attr()
        self._add_save_attr(
            _mask_data='mushroom'
        )
