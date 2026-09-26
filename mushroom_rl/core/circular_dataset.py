from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.dataset import Dataset
from mushroom_rl.core.dataset_info import DatasetInfo

from ._impl.layout import CodedLayout, RingLayout


class CircularDataset(Dataset):
    """
    Class for a dataset stored in a circular buffer of fixed capacity, whose new rows overwrite the oldest ones once
    it is full. No extra info is stored. Rows are indexed by buffer position.

    """
    def __init__(self, dataset_info, max_size):
        """
        Constructor.

        Args:
            dataset_info (DatasetInfo): the static information used to build the dataset;
            max_size (int): the capacity of the buffer.

        """
        super().__init__(dataset_info, n_steps=max_size)

    def __add__(self, other):
        result = self.copy()
        result += other

        return result

    def __iadd__(self, other):
        self.append_batch(other._time_ordered())

        return self

    def append_replay_batch(self, dataset):
        """
        Write a dataset at the write head, moving it forward. The dataset is converted to the backend and device of
        the buffer, and the episodes of joined datasets are reordered as by :meth:`Dataset.contiguous`. Its step
        information, episode information and policy parameters are dropped. A dataset starting with continuing rows
        continues the episodes left open by the previous write, one per open episode and in order; otherwise they are
        closed. An empty dataset writes nothing and leaves the open episodes open.

        Args:
            dataset (Dataset): the dataset to write.

        Returns:
            The buffer position of every row of ``dataset``, in its order, the positions of the open episode ends of
            the previous write that the dataset continues, and the positions of the stored steps whose previous step
            was overwritten.

        Raises:
            AssertionError: if ``dataset`` is a circular dataset;
            ValueError: if the dataset holds more rows than the buffer, or continues a number of episodes different
                from the number left open.

        """
        assert not dataset.is_circular, "Cannot append a circular dataset, join it with + or += instead."
        if len(dataset) > self._layout.max_size:
            raise ValueError(f"Cannot write {len(dataset)} rows to a buffer of {self._layout.max_size} rows.")
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        if len(dataset) == 0:
            nothing = backend.zeros(0, dtype=int, device=device)
            return nothing, list(), nothing

        dataset, order = dataset._glued()
        dataset = dataset.to_backend(self._dataset_info.env_backend, device=device)
        n = len(dataset)

        last = dataset._last_array()
        continues = dataset._layout.continues(last)
        start, positions, pairs, orphans = self._prepare_write(n, dataset._layout.pending_heads(),
                                                               dataset._layout.has_inner_break(last))

        self._write_rows(dataset)

        relinked = self._finish_write(start, positions, continues, pairs, dataset._layout.pending_tails(dataset.last))

        if order is not None:
            order = ArrayBackend.convert(order, to=self._dataset_info.env_backend, device=device)
            written = positions
            positions = backend.zeros(n, dtype=int, device=device)
            positions[order] = written

        return positions, relinked, orphans

    def append(self, step, info):
        """
        Write one step at the write head, moving it forward. The step continues the episode left open by the previous
        write, if any.

        Args:
            step (tuple): the transition, one entry per environment field followed by the policy state fields when
                the dataset is stateful;
            info (dict): the step information, dropped.

        Raises:
            ValueError: if the previous write left more than one episode open.

        """
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        heads = (0,) if len(self._layout.ring_tails) > 0 else tuple()
        start, positions, pairs, _ = self._prepare_write(1, heads, False)

        if self._layout.full:
            self._data[start] = step[:len(self._Field)]
            if self._agent_data is not None:
                self._agent_data[start] = step[len(self._Field):]
        else:
            self._store_step(step)
            self._layout.append()
        self._layout.advance(1)

        tails = tuple() if bool(step[self._Field.LAST]) else (0,)
        self._finish_write(start, positions, backend.zeros(0, dtype=bool, device=device), pairs, tails)

    def append_batch(self, other):
        """
        Write a dataset at the write head, as :meth:`append_replay_batch`.

        Args:
            other (Dataset): the dataset to write.

        Raises:
            AssertionError: if ``other`` is a circular dataset;
            ValueError: if the dataset holds more rows than the buffer, or continues a number of episodes different
                from the number left open.

        """
        self.append_replay_batch(other)

    def append_episode_info(self, info, mask=None):
        """
        Raises:
            TypeError: always, a circular dataset does not store episode information.

        """
        raise TypeError("A circular dataset does not store episode information.")

    def append_theta(self, theta):
        """
        Raises:
            TypeError: always, a circular dataset does not store policy parameters.

        """
        raise TypeError("A circular dataset does not store policy parameters.")

    def reserve(self, capacity):
        """
        Raises:
            NotImplementedError: always, the capacity of the buffer is fixed at construction.

        """
        raise NotImplementedError("The capacity of a circular dataset is fixed at construction.")

    def clear(self, history_context=None):
        """
        Drop every stored row, moving the write head back to the start of the buffer.

        Args:
            history_context (HistoryContext, None): unused; the rows written next continue no episode.

        """
        self._layout.clear()
        self._history_state = self._history_state.clear()
        self._clear_rows()

    def parse(self, to=None, device=None):
        """
        Return the stored episodes as a set of arrays, each episode from its oldest stored step on and the episodes
        oldest first. The returned ``last`` flags mark the final row of every stored segment.

        Args:
            to (str, None): the backend to be used for the returned arrays. By default, the dataset backend is used;
            device (str, None): device the returned arrays are placed on, or ``None`` for the default one.

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, reward, next state, absorbing
            and last.

        """
        return self._time_ordered().parse(to, device)

    def parse_policy_state(self, to=None, device=None):
        """
        Return the policy state arrays of the stored episodes, in the order of :meth:`parse`.

        Args:
            to (str, None): the backend to be used for the returned arrays. By default, the policy's backend is used;
            device (str, None): device the returned arrays are placed on, or ``None`` for the default one.

        Returns:
            A tuple containing the policy state and policy next state arrays.

        """
        return self._time_ordered().parse_policy_state(to, device)

    def to_backend(self, backend, device=None):
        """
        Return a copy of this dataset converted to the given backend.

        Args:
            backend (str): target backend (``'numpy'``, ``'torch'``, or ``'list'``);
            device (str, None): device the converted dataset is placed on, or ``None`` for the default one.

        Returns:
            A new CircularDataset with the capacity, the rows at their buffer positions and the write head of this
            one, in the requested backend, or ``self`` if the backend and the device already match.

        Raises:
            NotImplementedError: if converting to ``'list'`` a dataset holding policy states or history entries.

        """
        converted = super().to_backend(backend, device)
        if converted is self:
            return self

        dataset = type(self)(converted._dataset_info, self._layout.max_size)
        dataset._append_rows(converted)
        dataset._layout = converted._layout
        dataset._history_state = converted._history_state

        return dataset

    def history_cut(self, anchors, n_hops):
        """
        Check which anchors have a history window reaching a step that is no longer stored.

        Args:
            anchors (Array): the buffer position of each walk;
            n_hops (int): the number of steps to walk back.

        Returns:
            Whether walking back ``n_hops`` steps of the episode of each anchor reaches a step that is no longer stored.

        """
        return self._layout.history_cut(self._last_array(), anchors, n_hops)

    def select_first_episodes(self, n_episodes):
        """
        Return the oldest ``n_episodes`` stored episodes.

        Args:
            n_episodes (int): the number of episodes to pick from the dataset.

        Returns:
            A dataset containing the oldest ``n_episodes`` stored episodes, each in time order.

        Raises:
            IndexError: if the dataset holds fewer than ``n_episodes`` complete episodes.

        """
        return self._time_ordered().select_first_episodes(n_episodes)

    def get_init_states(self):
        """
        Get the initial states of the stored episodes.

        Returns:
            An array of the initial states of the stored episodes, oldest first.

        """
        return self._time_ordered().get_init_states()

    def compute_J(self, gamma=1., skip_incomplete=True):
        """
        Compute the cumulative discounted reward of each stored episode.

        Args:
            gamma (float, 1.): discount factor;
            skip_incomplete (bool, True): whether to leave out the episodes whose last step is not stored yet.

        Returns:
            The cumulative discounted reward of each stored episode, oldest first.

        """
        return self._time_ordered().compute_J(gamma, skip_incomplete)

    @classmethod
    def generate(cls, mdp_info, agent_info, max_size):
        """
        Build an empty circular dataset for the given MDP and agent.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            max_size (int): the capacity of the buffer.

        Returns:
            The empty circular dataset.

        """
        return cls(DatasetInfo.create_dataset_info(mdp_info, agent_info), max_size)

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_state=None, policy_next_state=None, extras=None,
                   horizon=None, gamma=0.99, backend='numpy', policy_backend=None, device=None, agent_device=None,
                   continuing=False, max_size=None):
        """
        Creates a circular dataset of transitions from the provided arrays, written in order from the start of the
        buffer.

        Args:
            states (array): array of states;
            actions (array): array of actions;
            rewards (array): array of rewards;
            next_states (array): array of next_states;
            absorbings (array): array of absorbing flags;
            lasts (array): array of last flags;
            policy_state (array, None): array of policy internal states;
            policy_next_state (array, None): array of next policy internal states;
            extras (ExtraInfo, None): step info, episode info and policy parameters, dropped;
            horizon (int, None): horizon of the mdp;
            gamma (float, 0.99): discount factor;
            backend (str, 'numpy'): backend to be used by the dataset;
            policy_backend (str, None): backend to be used for the policy state arrays; defaults to ``backend``;
            device (str, None): device the environment arrays are stored on, or ``None`` for the default one;
            agent_device (str, None): device the policy state arrays are stored on, or ``None`` for the default one;
            continuing (bool, False): whether row 0 continues an episode whose earlier steps are not stored;
            max_size (int, None): the capacity of the buffer; by default the number of transitions.

        Returns:
            The circular dataset.

        Raises:
            ValueError: if there are more transitions than ``max_size``.

        """
        dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                     policy_state=policy_state, policy_next_state=policy_next_state,
                                     horizon=horizon, gamma=gamma, backend=backend, policy_backend=policy_backend,
                                     device=device, agent_device=agent_device, continuing=continuing)
        circular = cls(dataset._dataset_info, len(dataset) if max_size is None else max_size)
        circular.append_replay_batch(dataset)

        return circular

    @property
    def episodes_length(self):
        """
        Compute the length of each stored episode.

        Returns:
            An array with the length of each stored episode, oldest first.

        """
        return self._time_ordered().episodes_length

    @property
    def is_circular(self):
        return True

    @property
    def max_size(self):
        """
        Returns:
            The capacity of the buffer.

        """
        return self._layout.max_size

    @property
    def write_head(self):
        """
        Returns:
            The buffer position the next row is written at.

        """
        return self._layout.write_head

    @property
    def full(self):
        """
        Returns:
            Whether the buffer has wrapped around.

        """
        return self._layout.full

    @property
    def size(self):
        """
        Returns:
            The number of rows stored.

        """
        return self._layout.size

    @property
    def links(self):
        """
        Returns:
            The ``(prev, next)`` arrays holding, for every buffer position, the distance to the previous and to the
            next step of its episode, 0 when there is none; ``None`` while every stored episode is written in
            consecutive positions, delimited by the stored ``last`` flags.

        """
        return self._layout.links

    def _prepare_write(self, n, heads, inner_break):
        max_size = self._layout.max_size
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        start = self._layout.write_head
        positions = (backend.arange(0, n, device=device) + start) % max_size

        to_close, pairs = self._layout.pair(heads)
        if len(to_close) > 0:
            self.last[to_close] = True

        adjacent = all((int(positions[head]) - tail) % max_size == 1 for tail, head in pairs)
        if self._layout.links is None and (not adjacent or inner_break):
            self._layout = self._layout.promote(self._last_array())

        orphans = self._layout.orphans(positions)

        return start, positions, pairs, orphans

    def _finish_write(self, start, positions, continues, pairs, tails):
        relinked = self._layout.link(positions, continues, pairs, start)
        self._layout.set_tails(tails, positions)

        return relinked

    def _write_rows(self, dataset):
        n_total = len(dataset)
        n = n_total
        max_size = self._layout.max_size
        head = self._layout.write_head

        if not self._layout.full:
            remaining = max_size - len(self)
            if n <= remaining:
                self._append_rows(dataset)
                self._layout.advance(n_total)
                return

            self._append_rows(dataset[:remaining])
            head = 0
            dataset = dataset[remaining:]
            n -= remaining

        first = min(n, max_size - head)
        self._overwrite_rows(head, dataset, slice(0, first))
        if n > first:
            self._overwrite_rows(0, dataset, slice(first, n))
        self._layout.advance(n_total)

    def _overwrite_rows(self, start, dataset, rows):
        stop = start + rows.stop - rows.start
        self._data[start:stop] = dataset._data[rows]
        if self._agent_data is not None:
            self._agent_data[start:stop] = dataset._agent_data[rows]

    def _time_ordered(self):
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        order, boundary_code = self._layout.time_order(self._last_array())
        tails = tuple()
        if len(self._layout.ring_tails) > 0:
            row_of = backend.zeros(self._layout.max_size, dtype=int, device=device)
            row_of[order] = backend.arange(0, len(order), device=device)
            tails = tuple(int(row_of[tail]) for tail in self._layout.ring_tails)
        layout = CodedLayout.from_array(boundary_code, self._dataset_info.env_backend, device, open_heads=tuple(),
                                        open_tails=tails)
        return self._view_rows(order, False, layout)

    def _walk_last(self):
        return self._last_array()

    def _create_layout(self, dataset_info, n_steps, n_envs):
        return RingLayout(dataset_info.env_backend, n_steps, dataset_info.env_device)

    @classmethod
    def _view_class(cls):
        return Dataset
