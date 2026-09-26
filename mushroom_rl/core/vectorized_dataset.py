from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.dataset import Dataset

from ._impl.containers import Container
from ._impl.layout import EpisodeLayout, CodedLayout
from ._impl.history_state import GridHistoryState


class VectorizedDataset(Dataset):
    """
    :class:`Dataset` variant for data collected from several environments in parallel. Each step stores a batch
    of transitions together with a boolean ``mask`` (kept in its own env-backend container) marking which
    environments were active. The steps are split off with :meth:`consume` and turned into a flat :class:`Dataset`
    with :meth:`flatten`, typically once per fit.

    """
    def __init__(self, dataset_info, n_steps=None, n_episodes=None, core_counts_episodes=False):
        super().__init__(dataset_info, n_steps, n_episodes, core_counts_episodes)

        env_backend = self._dataset_info.env_array_backend
        mask_shape = self._base_shape if self._base_shape is not None else ()
        self._mask_data = Container.create(dataset_info.env_backend, [mask_shape],
                                           [env_backend.to_backend_dtype(bool)],
                                           dataset_info.env_device, self._data.n_envs)
        self._tail_open = env_backend.zeros(self._data.n_envs, dtype=bool, device=dataset_info.env_device)
        self._consumed = False
        self._mask_shared = False
        self._history_state = GridHistoryState(self._data.n_envs, dataset_info.agent_backend,
                                               dataset_info.agent_device)

    def __add__(self, other):
        if len(other) == 0:
            return self.copy()
        if len(self) == 0:
            return other.copy()

        result = self.create_raw_instance(dataset=self)

        result._extras = self._extras + other._extras
        result._data = self._data + other._data
        result._agent_data = (self._agent_data + other._agent_data) if self._agent_data is not None else None
        result._layout = self._layout.join_rows(other._layout)
        result._mask_data = self._mask_data + other._mask_data
        result._history_state = self._history_state.copy()
        result._tail_open = self._dataset_info.env_array_backend.copy(self._tail_open)
        result._consumed = self._consumed or other._consumed

        return result

    def append(self, step, info):
        raise RuntimeError("Trying to use append on a vectorized dataset")

    def append_batch(self, other):
        if len(self) == 0:
            self._history_state = other._history_state.copy()
        self._append_rows(other)
        self._mask_data.append_batch(other._mask_data)
        self._consumed = self._consumed or other._consumed

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
        if self._dataset_info.env_backend == 'list':
            self._layout.append(self._dataset_info.env_array_backend.zeros(self._data.n_envs,
                                                                           dtype=self._boundary_dtype()))
        else:
            self._layout.append(int(EpisodeLayout.Boundary.NONE))
        self._extras.append_step(info)

    def append_theta_vectorized(self, theta, mask):
        """
        Append the policy parameters of the active environments to their per-environment ``theta`` lists.

        Args:
            theta (Array): the policy parameters, one entry per environment;
            mask (Array): boolean mask selecting the environments that are currently active.

        """
        self._extras.append_theta_vectorized(theta, mask)

    def consume(self, n_steps=None):
        """
        Split off the first ``n_steps`` active steps in collection (row-major) order, or every active step when
        ``n_steps`` is ``None``: they are marked inactive in this dataset's mask so that only the leftover remains
        active.

        Args:
            n_steps (int, None): number of steps to split off, at least the number of environments.

        Returns:
            A vectorized dataset sharing this one's data and holding only the steps split off.

        """
        assert n_steps is None or n_steps >= self._data.n_envs, \
            f"Cannot split off {n_steps} steps from {self._data.n_envs} environments."
        backend = self._dataset_info.env_array_backend
        mask = self.mask
        active = backend.where(mask.reshape(-1))[0]
        n_steps = len(active) if n_steps is None else n_steps
        assert 0 <= n_steps <= len(active)

        consumed_mask = backend.copy(mask)
        consumed_mask.reshape(-1)[active[n_steps:]] = False

        leftover_mask = backend.copy(mask)
        leftover_mask.reshape(-1)[active[:n_steps]] = False
        if self._mask_shared:
            self._mask_data = self._mask_data.get_view(slice(None), copy=True)
            self._mask_shared = False
        mask_column = self._mask_data.column()
        if isinstance(mask_column, list):
            mask_column[:] = list(leftover_mask)
        else:
            mask_column[:] = leftover_mask

        self._mark_heads(consumed_mask)

        view = self.create_raw_instance(dataset=self)
        view._extras = self._extras
        view._data = self._data
        view._agent_data = self._agent_data
        view._layout = self._layout
        view._history_state = self._history_state
        view._mask_data = self._allocate_mask(consumed_mask)
        view._tail_open = self._tail_open
        view._consumed = True

        return view

    def clear(self, keep_leftovers=False, history_context=None):
        """
        Clear the dataset. By default, the whole dataset is wiped. With ``keep_leftovers=True`` the steps still
        active after a :meth:`consume` (the leftover the fit did not consume) are compacted to the front and
        kept, so the next fit starts with them.

        Args:
            keep_leftovers (bool, False): whether to keep the leftover steps instead of wiping everything;
            history_context (HistoryContext, None): the history stream content around the most recent step; the
                first row of every environment in the next flat dataset that continues the cleared rows gets its
                window rebuilt from it.

        Returns:
            The number of steps kept.

        """
        backend = self._dataset_info.env_array_backend
        if keep_leftovers:
            row_active = backend.sum(self.mask, dim=1)
            n_carry = int(row_active.sum().item())

            if n_carry > 0:
                split_row = len(row_active) - int((row_active > 0).sum().item())
                self._data.compact(split_row)
                if self._agent_data is not None:
                    self._agent_data.compact(split_row)
                self._mask_data.compact(split_row)
                self._layout.compact(split_row)
                self._extras.keep_from(split_row)
                self._history_state.reset(history_context, self.mask[0], backend)

                return n_carry

        self._clear_rows()
        self._layout.clear()
        self._mask_data.clear()
        self._history_state.reset(history_context, backend.zeros(self._data.n_envs, dtype=bool,
                                                                 device=self._dataset_info.env_device), backend)

        return 0

    def flatten(self, n_steps=None):
        """
        Turn the padded per-environment data into a flat :class:`Dataset`, dropping the inactive entries via the
        mask and concatenating the environments end to end. A dataset returned by :meth:`consume` is flattened as
        it is; otherwise the first ``n_steps`` active steps are split off first, as :meth:`consume` does, or every
        active step when ``n_steps`` is ``None``.

        Args:
            n_steps (int, None): number of steps to split off, at least the number of environments.

        Returns:
            A flat :class:`Dataset`.

        """
        if self._consumed:
            assert n_steps is None, "The steps of a consumed dataset are already split off."
            return self._flatten()
        return self.consume(n_steps)._flatten()

    def get_view(self, index, copy=False):
        dataset = self._view_rows(index, copy, self._layout.view(index))
        dataset._mask_data = self._mask_data.get_view(index, copy)
        dataset._mask_shared = not copy
        dataset._tail_open = self._dataset_info.env_array_backend.copy(self._tail_open)
        dataset._consumed = self._consumed
        return dataset

    @classmethod
    def create_raw_instance(cls, dataset=None):
        new_dataset = super().create_raw_instance(dataset)

        new_dataset._mask_data = None
        new_dataset._mask_shared = False
        new_dataset._tail_open = None
        new_dataset._consumed = False

        return new_dataset

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

    def _create_layout(self, dataset_info, n_steps, n_envs):
        boundary_shape = self._base_shape if self._base_shape is not None else ()
        return CodedLayout(dataset_info.env_backend, boundary_shape, dataset_info.env_device, n_envs)

    def _allocate_mask(self, mask):
        rows = Container.from_array([mask], device=self._dataset_info.env_device,
                                    backend=self._dataset_info.env_backend)
        capacity = self._data.capacity
        if capacity is None:
            return rows
        container = Container.create(self._dataset_info.env_backend, [(capacity,) + tuple(mask.shape[1:])],
                                     [self._dataset_info.env_array_backend.to_backend_dtype(bool)],
                                     self._dataset_info.env_device, self._data.n_envs)
        container.append_batch(rows)
        return container

    def _mark_heads(self, consumed_mask):
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        active = backend.sum(consumed_mask, dim=0) > 0
        if bool(active.any()):
            n_rows = len(consumed_mask)
            steps = backend.expand_dims(backend.arange(0, n_rows, device=device), 1)
            first_row = backend.min(backend.where(consumed_mask, steps, n_rows), dim=0)[active]
            last_row = backend.max(backend.where(consumed_mask, steps, -1), dim=0)[active]
            envs = backend.arange(0, len(active), device=device)[active]
            kinds = backend.zeros(len(active), dtype=self._boundary_dtype(), device=device)
            kinds[:] = int(EpisodeLayout.Boundary.FRESH)
            kinds[self._tail_open] = int(EpisodeLayout.Boundary.CONTINUING)

            column = self._layout.column()
            if isinstance(column, list):
                grid = backend.as_array(column, device=device)
                grid[first_row, envs] = kinds[active]
                column[:] = list(grid)
            else:
                column[first_row, envs] = kinds[active]

            self._tail_open[active] = ~(self._last_array()[last_row, envs] > 0)

    def _flatten(self):
        if len(self) == 0:
            return Dataset(self._dataset_info.flat(), n_steps=0)

        mask = self.mask
        env_backend = self._dataset_info.env_array_backend
        agent_backend = self._dataset_info.agent_array_backend
        device = self._dataset_info.env_device

        states = env_backend.pack_padded_sequence(self.state, mask)
        actions = env_backend.pack_padded_sequence(self.action, mask)
        rewards = env_backend.pack_padded_sequence(self.reward, mask)
        next_states = env_backend.pack_padded_sequence(self.next_state, mask)
        absorbings = env_backend.pack_padded_sequence(self.absorbing, mask)
        lasts = env_backend.pack_padded_sequence(self._last_array(), mask)

        flags = ArrayBackend.get_array_backend_from(mask)
        boundary = flags.as_array(env_backend.pack_padded_sequence(self._layout.array(), mask), device=device)
        counts = flags.sum(mask, dim=0)
        env_of_row = flags.repeat(flags.arange(0, len(counts), device=device), counts)
        same_env = env_of_row[1:] == env_of_row[:-1]
        continuing = boundary[1:] & int(EpisodeLayout.Boundary.CONTINUING) > 0
        boundary[1:][continuing & same_env] = int(EpisodeLayout.Boundary.NONE)
        env_end = flags.concatenate([~same_env, flags.ones(1, dtype=bool, device=device)])
        open_end = env_end & ~(flags.as_array(lasts, device=device) > 0)
        open_heads = tuple(int(i) for i in flags.where(boundary & int(EpisodeLayout.Boundary.CONTINUING) > 0)[0])
        open_tails = tuple(int(i) for i in flags.where(open_end)[0])

        policy_state = None
        policy_next_state = None

        if self.is_stateful:
            policy_mask = agent_backend.convert_mask(mask, backend=env_backend, device=self._dataset_info.agent_device)
            policy_state, policy_next_state = self.policy_state, self.policy_next_state
            if self._dataset_info.env_backend == 'list':
                agent_device = self._dataset_info.agent_device
                policy_state = agent_backend.from_list(policy_state, device=agent_device)
                policy_next_state = agent_backend.from_list(policy_next_state, device=agent_device)
            policy_state = agent_backend.pack_padded_sequence(policy_state, policy_mask)
            policy_next_state = agent_backend.pack_padded_sequence(policy_next_state, policy_mask)

        flat_extras = self._extras.flatten(mask)

        layout = CodedLayout.from_array(boundary, env_backend.get_backend_name(), device, open_heads=open_heads,
                                        open_tails=open_tails)
        flat = Dataset._from_components(states, actions, rewards, next_states, absorbings, lasts, layout,
                                        policy_state=policy_state, policy_next_state=policy_next_state,
                                        extras=flat_extras, horizon=self._dataset_info.horizon,
                                        gamma=self._dataset_info.gamma,
                                        backend=env_backend.get_backend_name(),
                                        policy_backend=agent_backend.get_backend_name(),
                                        device=device,
                                        agent_device=self._dataset_info.agent_device)
        positions, is_episode_start = flat._segment_starts(flat._last_array())
        flat._history_state = self._history_state.emit(positions, env_of_row[positions], is_episode_start, flags)

        return flat

    def _add_all_save_attr(self):
        super()._add_all_save_attr()
        serialization = self._dataset_info.env_array_backend.get_backend_serialization() \
            if self._dataset_info is not None else 'pickle'
        self._add_save_attr(
            _mask_data='mushroom',
            _tail_open=serialization,
            _consumed='primitive',
            _mask_shared='primitive'
        )
