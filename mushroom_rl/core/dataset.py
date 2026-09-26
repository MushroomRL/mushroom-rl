import numpy as np
import math

from enum import IntEnum

import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.dataset_info import DatasetInfo
from mushroom_rl.core.extra_info import ExtraInfo

from ._impl.containers import Container
from ._impl.layout import EpisodeLayout, StreamLayout
from ._impl.history_state import HistoryState

from mushroom_rl.utils.episodes import split_episodes


class Dataset(MushroomObject):
    """
    Collection of the transitions gathered while an agent interacts with an environment. The data is split into
    two backend-aware groups:

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
        self._data = Container.create(dataset_info.env_backend, env_shapes, env_dtypes, dataset_info.env_device,
                                      n_envs)

        if dataset_info.policy_state_shape is not None:
            policy_shapes, policy_dtypes = self._policy_specs(dataset_info, self._base_shape)
            # the policy state lives in the agent backend, but a list (infinite-horizon) env keeps it growable
            agent_backend = 'list' if dataset_info.env_backend == 'list' else dataset_info.agent_backend
            self._agent_data = Container.create(agent_backend, policy_shapes, policy_dtypes,
                                                dataset_info.agent_device, n_envs)
        else:
            self._agent_data = None

        self._layout = self._create_layout(dataset_info, n_steps, n_envs)
        self._history_state = HistoryState(dataset_info.agent_backend, dataset_info.agent_device)

        self._add_all_save_attr()

    def __getitem__(self, index):
        if isinstance(index, (slice, np.ndarray, torch.Tensor)):
            return self.get_view(index)
        elif isinstance(index, int) and index < len(self._data):
            return self._data[index]
        else:
            raise IndexError

    def __add__(self, other):
        other = other._time_ordered()
        if len(other) == 0:
            return self.copy()
        if len(self) == 0:
            return other.copy()

        layout, stitched = self._layout.concatenate(other._layout, self.last)

        result = self.create_raw_instance(dataset=self)

        result._extras = self._extras + other._extras
        result._data = self._data + other._data
        result._agent_data = (self._agent_data + other._agent_data) if self._agent_data is not None else None
        result._layout = layout
        result._history_state = self._history_state.concatenate(other._history_state, len(self), stitched)

        return result

    def __iadd__(self, other):
        other = other._time_ordered()
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other.copy()

        capacity = self._data.capacity
        if capacity is not None and len(self) + len(other) > capacity:
            return self + other

        self.append_batch(other)
        self._extras += other._extras

        return self

    def __len__(self):
        return len(self._data)

    def append(self, step, info):
        self._layout.append()
        self._store_step(step)
        self._extras.append_step(info)

    def append_batch(self, other):
        """
        Append the transitions of another dataset in place without copying data. Only the transition data is
        merged (env data and policy state); the step information, the per-episode information and the policy
        parameters are not updated.

        Args:
            other (Dataset): dataset whose transitions will be appended.

        Raises:
            AssertionError: if ``other`` is a circular dataset.

        """
        assert not other.is_circular, "Cannot append a circular dataset, join it with + or += instead."
        n = len(self)
        self._layout, stitched = self._layout.append_batch(other._layout, self.last)
        self._history_state = self._history_state.concatenate(other._history_state, n, stitched)
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
        self._layout.reserve(capacity)

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

    def clear(self, history_context=None):
        """
        Clear the dataset.

        Args:
            history_context (HistoryContext, None): the history stream content around the most recent step; when the
                next row continues the cleared ones, its window is rebuilt from it.

        """
        self._layout.clear(self.last)
        if history_context is not None and self._layout.first == EpisodeLayout.Boundary.CONTINUING:
            self._history_state = HistoryState.from_context(self._dataset_info.agent_backend,
                                                            self._dataset_info.agent_device, history_context)
        else:
            self._history_state = self._history_state.clear()
        self._clear_rows()

    def get_view(self, index, copy=False):
        """
        Select a subset of the transitions. The selection is a new standalone dataset: it does not continue any
        episode of the original one. A contiguous subset keeps the episode structure of the selected transitions.

        Args:
            index (slice or Array): the transitions to select;
            copy (bool, False): whether the returned dataset owns a copy of the selected data.

        Returns:
            The dataset of the selected transitions.

        """
        dataset = self._view_rows(index, copy, self._layout.standalone_view(index, self._last_array()))
        dataset._history_state = HistoryState(self._dataset_info.agent_backend, self._dataset_info.agent_device)

        return dataset

    def item(self):
        assert len(self) == 1
        return self[0]

    def parse(self, to=None, device=None):
        """
        Return the dataset as a set of arrays. The returned ``last`` flags mark the final row of every stored
        segment: every episode end, every row whose successor is not stored right after it, and the final row.

        Args:
            to (str, None):  the backend to be used for the returned arrays. By default, the dataset backend is used;
            device (str, None): device the returned arrays are placed on, or ``None`` for the default one.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, reward, next state, absorbing
            and last.

        """
        if to is None:
            to = self._dataset_info.env_array_backend.get_backend_name()
        return self._convert(self.state, self.action, self.reward, self.next_state, self.absorbing,
                             self.last_or_boundary, to=to, device=device)

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
        Return this dataset converted to the given backend. The converted dataset may share memory with this one.

        Args:
            backend (str): target backend (``'numpy'``, ``'torch'``, or ``'list'``);
            device (str, None): device the converted dataset is placed on, or ``None`` for the default one.

        Returns:
            A new Dataset in the requested backend, or ``self`` if the backend and the device already match.

        Raises:
            NotImplementedError: if converting to ``'list'`` a dataset holding policy states or history entries.

        """
        if self._same_backend(self._dataset_info.env_array_backend, self._dataset_info.env_device, backend, device) \
                and self._same_backend(self._dataset_info.agent_array_backend, self._dataset_info.agent_device,
                                       backend, device):
            return self
        if backend == 'list' and (self.is_stateful or len(self._history_state) > 0):
            raise NotImplementedError("Converting policy states or history entries to the list backend is not "
                                      "currently supported.")
        state, action, reward, next_state, absorbing, last = self._convert(
            self.state, self.action, self.reward, self.next_state, self.absorbing, self.last, to=backend, device=device)
        policy_state, policy_next_state = (self._convert(self.policy_state, self.policy_next_state, to=backend,
                                                         backend=self._dataset_info.agent_array_backend, device=device)
                                           if self.is_stateful else (None, None))
        return Dataset._from_components(state, action, reward, next_state, absorbing, last,
                                        self._layout.to_backend(backend, device),
                                        policy_state=policy_state, policy_next_state=policy_next_state,
                                        extras=self._extras.to_backend(backend, device),
                                        horizon=self._dataset_info.horizon, gamma=self._dataset_info.gamma,
                                        backend=backend, policy_backend=backend, device=device, agent_device=device,
                                        history_state=self._history_state.to_backend(backend, device))

    def walk_back(self, anchors, n_hops, last=None):
        """
        Walk back from each anchor along its episode.

        Args:
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk;
            last (None): the flags of the final row of every stored segment; by default the ones of this dataset. The
                walk runs on their backend and device.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step is stored. A missing step holds the
            last row reached.

        """
        return self._layout.walk_back(self._walk_last() if last is None else last, anchors, n_hops)

    def walk_forward(self, anchors, n_hops, last=None):
        """
        Walk forward from each anchor along its episode.

        Args:
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk;
            last (None): the flags of the final row of every stored segment, as in :meth:`walk_back`.

        Returns:
            The ``(positions, valid)`` arrays, as in :meth:`walk_back`.

        """
        return self._layout.walk_forward(self._walk_last() if last is None else last, anchors, n_hops)

    def contiguous(self):
        """
        Reorder the rows so that every episode continued across a join (``+`` or ``+=``) follows its past rows. The
        open episodes of each joined dataset pair, in order, with the continuing episodes of the next one; the other
        rows keep their order.

        Returns:
            The reordered dataset, or this dataset when no join has episodes to pair.

        Raises:
            ValueError: if a joined dataset continues a number of episodes different from the number left open before
            it.

        """
        dataset, _ = self._glued()
        return dataset

    def select_first_episodes(self, n_episodes):
        """
        Return the first ``n_episodes`` episodes in the provided dataset.

        Args:
            n_episodes (int): the number of episodes to pick from the dataset;

        Returns:
            A subset of the dataset containing the first ``n_episodes`` episodes.

        Raises:
            IndexError: if the dataset holds fewer than ``n_episodes`` complete episodes.

        """
        assert n_episodes > 0, 'Number of episodes must be greater than zero.'

        backend = self._dataset_info.env_array_backend
        last = self._last_array()
        ends = self._segment_ends(last) > 0
        flags = ends * 1
        segment = backend.cumsum(flags) - flags
        complete = (last[backend.where(ends)[0]] > 0) * 1
        n_complete = int(backend.sum(complete))
        if n_complete < n_episodes:
            raise IndexError(f"The dataset holds {n_complete} complete episodes, {n_episodes} were requested.")
        kept = (complete > 0) & (backend.cumsum(complete) <= n_episodes)
        rows = backend.where(kept[segment])[0]
        if int(rows[-1]) == len(rows) - 1:
            return self[:len(rows)]
        return self._view_episodes(rows)

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
        positions, is_episode_start = self._segment_starts(self._last_array())
        x_0 = [self.state[int(i)] for i in positions[is_episode_start]]
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

        last = self._last_array()
        reward = backend.as_array(self.reward, device=device)

        if len(last) == 0:
            return backend.zeros(0, device=device)

        ends_ep, r_ep, last_ep = split_episodes(self._segment_ends(last), reward, last)

        if len(r_ep.shape) == 1:
            ends_ep, r_ep, last_ep = (backend.expand_dims(array, 0) for array in (ends_ep, r_ep, last_ep))
        if skip_incomplete:
            r_ep = r_ep[last_ep[ends_ep > 0] > 0]
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
        new_dataset._layout = None
        new_dataset._history_state = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_state=None, policy_next_state=None, extras=None,
                   horizon=None, gamma=0.99, backend='numpy', policy_backend=None, device=None, agent_device=None,
                   continuing=False):
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
            agent_device (str, None): device the policy state arrays are stored on, or ``None`` for the default one;
            continuing (bool, False): whether row 0 continues the episode left open at the end of the dataset this
                one is appended to. By default the dataset is standalone: consecutive rows are consecutive steps of
                one stream and row 0 continues nothing.

        Returns:
            The list of transitions.

        """
        layout = StreamLayout.from_rows(len(states), backend, device, continuing=continuing)
        return cls._from_components(states, actions, rewards, next_states, absorbings, lasts, layout,
                                    policy_state=policy_state, policy_next_state=policy_next_state, extras=extras,
                                    horizon=horizon, gamma=gamma, backend=backend, policy_backend=policy_backend,
                                    device=device, agent_device=agent_device)

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
    def last_or_boundary(self):
        """
        The flags marking the final row of every stored segment: every ``last`` plus every row whose successor is not
        stored right after it, and the final row.

        """
        ends = self._segment_ends(self._last_array())
        if self._dataset_info.env_backend == 'list':
            return list(ends)
        return ends

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
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device

        last = self._last_array()
        if len(last) == 0:
            return backend.as_array(list(), device=device)

        end = backend.where(self._segment_ends(last) > 0)[0]
        start = backend.concatenate([backend.zeros(1, dtype=int, device=device), end[:-1] + 1])
        return (end - start + 1)[last[end] > 0]

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
    def is_circular(self):
        """
        Whether the dataset is a circular buffer.

        """
        return False

    @property
    def capacity(self):
        """
        The number of transitions the dataset can hold, or ``None`` when it grows without bound (list backend).

        """
        return self._data.capacity

    @property
    def history_state(self):
        """
        The history stream entries attached to the rows that start a segment continuing rows stored elsewhere.

        """
        return self._history_state

    def _agent_rows(self, rows):
        if isinstance(rows, (int, slice)):
            return rows
        return ArrayBackend.convert(rows, to=self._dataset_info.agent_backend, device=self._dataset_info.agent_device)

    def _last_array(self):
        return self._dataset_info.env_array_backend.as_array(self.last, device=self._dataset_info.env_device)

    def _boundary_dtype(self):
        return self._dataset_info.env_array_backend.to_backend_dtype(np.int8)

    def _segment_ends(self, last):
        return self._layout.segment_ends(last)

    def _walk_last(self):
        return self._segment_ends(self._last_array())

    def _time_ordered(self):
        return self

    def _glued(self):
        if self._layout.n_joins == 0:
            return self, None
        order, layout, glued = self._layout.glue(self._last_array())
        dataset = self.create_raw_instance(dataset=self)
        dataset._extras = self._extras.reorder_steps(order)
        dataset._data = self._data.get_view(order, copy=True)
        dataset._agent_data = (self._agent_data.get_view(self._agent_rows(order), copy=True)
                               if self._agent_data is not None else None)
        dataset._layout = layout
        dataset._history_state = self._history_state.drop(glued, len(self)).get_view(order, len(self))
        return dataset, order

    def _create_layout(self, dataset_info, n_steps, n_envs):
        boundary_shape = self._base_shape if self._base_shape is not None else ()
        return StreamLayout(dataset_info.env_backend, boundary_shape, dataset_info.env_device, n_envs)

    def _segment_starts(self, last):
        return self._layout.segment_starts(last)

    def _view_rows(self, index, copy, layout):
        dataset = self._view_class().create_raw_instance(dataset=self)

        dataset._extras = self._extras.get_view(index, copy)
        dataset._data = self._data.get_view(index, copy)
        dataset._agent_data = (self._agent_data.get_view(self._agent_rows(index), copy)
                               if self._agent_data is not None else None)
        dataset._layout = layout
        dataset._history_state = self._history_state.get_view(index, len(self))

        return dataset

    def _clear_rows(self):
        self._extras.clear()
        self._data.clear()
        if self._agent_data is not None:
            self._agent_data.clear()

    def _view_episodes(self, rows):
        dataset = self._view_rows(rows, False, self._layout.episodes_view(rows, self._last_array()))
        dataset._history_state = HistoryState(self._dataset_info.agent_backend, self._dataset_info.agent_device)
        return dataset

    def _append_rows(self, other):
        self._data.append_batch(other._data)
        if self._agent_data is not None:
            self._agent_data.append_batch(other._agent_data)
        self._layout = self._layout.append_rows(other._layout)

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
            _layout='mushroom',
            _history_state='mushroom',
            _base_shape='primitive',
            _dataset_info='mushroom'
        )

    @classmethod
    def _view_class(cls):
        return cls

    @classmethod
    def _from_components(cls, states, actions, rewards, next_states, absorbings, lasts, layout,
                         policy_state=None, policy_next_state=None, extras=None, horizon=None, gamma=0.99,
                         backend='numpy', policy_backend=None, device=None, agent_device=None, history_state=None):
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(absorbings) == len(lasts)

        if policy_state is not None:
            assert len(states) == len(policy_state) == len(policy_next_state)

        if policy_backend is None:
            policy_backend = backend

        dataset = cls.create_raw_instance()

        dataset._extras = ExtraInfo(1, backend) if extras is None else extras.copy()

        env_arrays = [states, actions, rewards, next_states, absorbings, lasts]
        dataset._data = Container.from_array(env_arrays, device=device, backend=backend)

        if policy_state is not None:
            agent_arrays = [policy_state, policy_next_state]
            dataset._agent_data = Container.from_array(agent_arrays, device=agent_device, backend=policy_backend)
        else:
            dataset._agent_data = None

        dataset._layout = layout
        dataset._history_state = HistoryState(policy_backend, agent_device) if history_state is None else history_state

        state_shape = cls._infer_shape(states)
        action_shape = cls._infer_shape(actions)
        state_dtype = cls._infer_dtype(states)
        action_dtype = cls._infer_dtype(actions)
        policy_state_shape = None if policy_state is None else cls._infer_shape(policy_state)

        dataset._dataset_info = DatasetInfo(backend, policy_backend, device, agent_device, horizon, gamma,
                                            state_shape, state_dtype, action_shape, action_dtype, policy_state_shape)

        return dataset

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
    def _infer_shape(data):
        if hasattr(data, 'shape'):
            return data.shape[1:]
        return data[0].shape if len(data) and hasattr(data[0], 'shape') else ()

    @staticmethod
    def _infer_dtype(data):
        if hasattr(data, 'dtype'):
            return data.dtype
        return data[0].dtype if len(data) and hasattr(data[0], 'dtype') else None

    @staticmethod
    def _same_backend(array_backend, stored_device, backend, device):
        return array_backend.get_backend_name() == backend \
            and array_backend.check_device(device) == array_backend.check_device(stored_device)
