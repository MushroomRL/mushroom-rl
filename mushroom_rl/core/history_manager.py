from mushroom_rl.core.mushroom_backend_object import MushroomBackendObject
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.batch import Batch

from mushroom_rl.core._impl.layout import EpisodeLayout
from mushroom_rl.core._impl.history_state import HistoryContext


class HistoryManager(MushroomBackendObject):
    """
    Object in charge of assembling the per-timestep context fed to the policy: the observation is preprocessed and
    then stacked with the most recent entries of one or more streams.

    The manager owns the agent preprocessors and applies them wherever an observation becomes policy input,
    online in :meth:`__call__` and offline in the ``parse_*`` methods, so the two cannot disagree. Preprocessing
    happens before the stacking, hence the zero padding of a window shorter than its stream stays zero. The
    statistics are advanced only by :meth:`update_preprocessors`, on the flat observation stream of a dataset.

    The context is a deterministic function of the observed trajectory, hence it is always reconstructable from the
    stored transitions and is not part of the (latent) policy state. The manager holds an ordered set of named streams,
    each with its own stacking length and an ``offset`` telling how many steps behind the current one its window ends
    (0 for the observation, 1 for the previous action). The two reserved streams are sourced by the manager itself: the
    ``obs_history`` stream from the ``state`` passed to :meth:`__call__` (delivered in-band as the policy ``state``)
    and the ``action_history`` stream from the last action recorded through :meth:`record_action`. Any further stream
    is fed as a keyword argument to :meth:`__call__` and returned under its own name. With no active stream the manager
    is the identity context: it passes the ``state`` through unchanged and returns no keyword arguments, so an agent
    that does no stacking can hold one unconditionally instead of a ``None``.

    The manager works entirely in the agent backend. Online it stacks the most recent entry of each stream, and the
    same stacking rule is exposed offline through :meth:`build_history`, for every row of a buffer or for a batch of
    anchors, so the window built while interacting with the environment and the one rebuilt from a stored buffer are
    guaranteed to match. Each per-step window is ``(length, *shape)``, squeezed to ``(*shape)`` when ``length`` is 1.

    Each stream is described by a specification dictionary with the keys ``length``, ``shape`` and ``dtype`` plus any
    number of options; the only option acted upon by this class is ``offset`` (default 0). Subclasses may store and
    honor richer options without changing the base machinery. The reserved streams and their conventional
    lengths/offsets are wired up from the MDP and action spaces by the :meth:`default_streams` factory.

    """
    _actor_selection = None

    def __init__(self, mdp_info, agent_info, streams=None, preprocessors=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP, providing the observation and action spaces the reserved
                streams are checked against;
            agent_info (AgentInfo): information about the agent, providing the array backend and the device on which
                the manager keeps its buffers;
            preprocessors (list, None): preprocessors applied to every observation before it is stacked, both online
                in :meth:`__call__` and offline in the ``parse_*`` methods;
            streams (dict, None): the named streams assembled by the manager, given as a mapping ``name -> spec``,
                where each ``spec`` is a dictionary with the keys ``length``, ``shape`` and ``dtype`` and, optionally,
                ``offset`` (default 0), the number of steps behind the current one at which the window ends. Each spec
                is forwarded to :meth:`add_stream`, and each stream's window is returned under ``name`` in the output
                of :meth:`__call__`. The two reserved names are sourced by the manager itself: ``obs_history`` (from
                the in-band ``state``, returned positionally) and ``action_history`` (from the last recorded action,
                conventionally ``offset`` 1). These reserved streams, with their shapes and data types read from the
                MDP and action spaces, are wired up by :meth:`default_streams`.

        Raises:
            AssertionError: if the entry shape or data type of a reserved stream differs from the one of its space.

        """
        super().__init__(agent_info.backend_name, agent_info.device)

        self._stream_specs = dict()
        self._preprocessors = list(preprocessors) if preprocessors else list()
        self._buffers = None
        self._last_action = None
        self._last_windows = dict()
        self._n_envs = None

        self._add_save_attr(
            _stream_specs='primitive',
            _preprocessors='mushroom',
            _buffers='none',
            _last_action='none',
            _last_windows='none',
            _n_envs='none'
        )

        for name, spec in (streams or dict()).items():
            self.add_stream(name, **spec)

        self._check_reserved_streams(mdp_info)

    def __call__(self, state=None, **extra):
        """
        Append the current entries to the buffers and return the per-timestep context split for the policy call: the
        observation input to be passed positionally as ``state`` and a dictionary of the additional conditioning
        streams to be forwarded as keyword arguments. The observation is preprocessed before being stacked. The
        reserved ``obs_history`` stream, when active, replaces ``state`` with its stacked window; otherwise the
        preprocessed ``state`` is passed through unstacked. The
        ``action_history`` stream is sourced from the last action recorded through :meth:`record_action`. Each
        remaining stream is forwarded under its own name.

        Args:
            state: the current observation, already in the agent backend, consumed by the observation stream;
            **extra: the current value of each other stream, keyed by its name and already in the agent
                backend; a stream whose value is not provided is zero-padded for that step.

        Returns:
            A tuple ``(state, policy_kwargs)`` ready to be used as ``policy.draw_action(state, **policy_kwargs)``. Each
            window has shape ``(length, *shape)`` (single-environment) or ``(n_envs, length, *shape)`` (vectorized),
            squeezed along the ``length`` axis when the stream length is 1.

        """
        obs = None if state is None else self.preprocess(state)
        streams = self._observation_streams()
        entries = {name: None if obs is None else self._select(obs, selection) for name, selection in streams.items()
                   if name in self._stream_specs}

        windows = dict()
        for name in self._stream_specs:
            if name in entries:
                value = entries[name]
            elif name == 'action_history':
                value = self._last_action
            else:
                value = extra.get(name)
            windows[name] = self._stack(name, value)
        self._last_windows = dict(windows)

        if 'obs_history' in windows:
            state = windows['obs_history']
        elif obs is not None:
            state = self._select(obs, streams['obs_history'])
        return state, {name: window for name, window in windows.items() if name not in entries}

    def add_stream(self, name, length, shape, dtype, offset=0, **options):
        """
        Register a named stream to be stacked by the manager. Streams are usually declared through the constructor, but
        this method allows building a manager and adding buffers programmatically.

        Args:
            name (str): the name under which the stream's window is returned by :meth:`__call__` (``obs_history`` is
                reserved for the in-band observation stream);
            length (int): number of entries stacked in the stream's window; may be 1 only at a non-zero ``offset``;
            shape (tuple): shape of a single entry of the stream;
            dtype: data type of the stream, converted to the agent backend;
            offset (int, 0): number of steps behind the current one at which the window ends, at most 1;
            **options: additional per-stream options stored in the specification; ignored by the base class.

        Raises:
            AssertionError: if ``offset`` is greater than 1, or if the stream stacks nothing (``length`` 1 at
                ``offset`` 0);
            NotImplementedError: if the agent backend is ``'list'``.

        """
        if self.backend_name == 'list':
            raise NotImplementedError("History stacking is not currently supported with the list agent backend.")
        assert length > 1 or offset > 0, "A stream of length 1 at offset 0 returns the current entry unchanged. " \
                                         "Leave it unregistered instead."
        assert offset <= 1, "A stream offset greater than 1 is not supported: the manager retains no entry older " \
                            "than the ones its window holds, plus the current one."

        self._stream_specs[name] = dict(length=length, shape=tuple(shape),
                                        dtype=self._backend.to_backend_dtype(dtype), offset=offset, **options)

    def add_preprocessor(self, preprocessor):
        """
        Append a preprocessor to the list applied to every observation before it is stacked. The preprocessors are
        applied in the order they are added.

        Args:
            preprocessor (Preprocessor): the preprocessor to apply to the observations, operating in the agent backend.

        """
        backend_name = self.backend_name
        assert preprocessor.backend_name == backend_name, \
            f"The preprocessor operates on '{preprocessor.backend_name}' arrays, but the observations reaching it " \
            f"are in the agent backend '{backend_name}'. Build it with backend='{backend_name}'."
        assert self._backend.same_device(preprocessor.device, self._device), \
            f"The preprocessor is on '{preprocessor.device}', but the observations reaching it are on the agent " \
            f"device '{self._device}'. Build it with device='{self._device}'."

        self._preprocessors.append(preprocessor)

    def reset(self):
        """
        Reset the buffers at the beginning of a single-environment episode.

        """
        if self._n_envs is not None or self._buffers is None:
            self._allocate_buffers(None)
        else:
            self._zero_buffers()

    def reset_vectorized(self, start_mask):
        """
        Reset the buffers for the environments selected by ``start_mask``, leaving the others untouched. The buffers
        are (re)allocated the first time this is called and whenever the number of environments changes; otherwise the
        selected environments are zeroed in place.

        Args:
            start_mask: boolean mask selecting the environments that are starting a new episode.

        """
        n_envs = len(start_mask)
        if self._n_envs != n_envs:
            self._allocate_buffers(n_envs)
        else:
            self._zero_buffers_vectorized(start_mask)

    def record_action(self, action):
        """
        Record the action just drawn by the agent so that it becomes the most recent entry of the ``action_history``
        window at the next step (its ``offset`` 1). Called by the agent after every ``draw_action``. The last action
        is always kept (it is cheap and available for custom logging), even when the previous-action stream is not
        active, and it is not stacked into any window.

        Args:
            action: the action just drawn, already in the agent backend.

        """
        self._last_action = action

    def preprocess(self, obs):
        """
        Apply every preprocessor, in order, to the given observations. Per-observation parameters broadcast over any
        leading axes, so the same call works on a single observation, on a batch and on a stack of windows.

        Args:
            obs: the observations to preprocess, already in the agent backend.

        Returns:
            The preprocessed observations.

        """
        for p in self._preprocessors:
            obs = p(obs)
        return obs

    def update_preprocessors(self, dataset):
        """
        Update the statistics of every preprocessor from the observations of a dataset, counting each of them once.
        Takes the dataset rather than an array so that the flat observation stream, and never a stacked window, is
        used for the update.

        Args:
            dataset (Dataset): the dataset whose observations update the statistics.

        """
        state = self._backend.convert(dataset.state, device=self._device)
        for i, p in enumerate(self._preprocessors, 1):
            p.update(state)
            if i < len(self._preprocessors):
                state = p(state)

    def parse_state(self, dataset, to=None):
        """
        Rebuild the preprocessed observation windows of a dataset, i.e. the ``state`` of :meth:`parse_history` without
        also rebuilding the next-state and the other stream windows.

        Args:
            dataset (Dataset): the dataset to parse;
            to (str, None): the backend of the returned array; when ``None`` the agent backend is used.

        Returns:
            The stacked observation windows, preprocessed. A stream stacking a single entry collapses to the raw
            value.

        """
        if 'obs_history' in self._stream_specs:
            states, last = self._backend.convert(dataset.state, dataset.last_or_boundary, device=self._device)
            state = self.build_history('obs_history', states, last, attachment=dataset.history_state)
        else:
            state = self._backend.convert(dataset.state, device=self._device)
            state = self._select(self.preprocess(state), self._actor_selection)

        return self._convert_output(to, state)

    def parse_initial_state(self, dataset, to=None):
        """
        Build the preprocessed observation window of each episode start of a dataset, i.e. :meth:`parse_state`
        restricted to the observations returned by :meth:`~mushroom_rl.core.dataset.Dataset.get_init_states`. These
        windows have no history behind them, so the older entries are zero-padded.

        Args:
            dataset (Dataset): the dataset to parse;
            to (str, None): the backend of the returned array; when ``None`` the agent backend is used.

        Returns:
            The stacked observation windows of the initial states, preprocessed. A stream stacking a single entry
            collapses to the raw value.

        """
        states = self._backend.convert(dataset.get_init_states(), device=self._device)

        if 'obs_history' in self._stream_specs:
            last = self._backend.ones(len(states), dtype=bool, device=self._device)
            state = self.build_history('obs_history', states, last)
        else:
            state = self._select(self.preprocess(states), self._actor_selection)

        return self._convert_output(to, state)

    def parse_history(self, dataset, anchor_idxs=None, to=None, get_policy_state=False):
        """
        Parse a dataset into a batch with the history windows applied, as :meth:`Dataset.parse`: ``state`` and
        ``next_state`` are the policy input of the current and of the next step, ``critic_state`` and
        ``critic_next_state`` the critic input when it differs from the policy one, and ``action_history`` and
        ``next_action_history`` the previous-action windows of the current and of the next step when that stream is
        active.

        Args:
            dataset (Dataset): the dataset to parse;
            anchor_idxs (None): the rows to parse, buffer positions for a circular dataset; by default every row;
            to (str, None): the backend of the returned arrays; by default the agent backend;
            get_policy_state (bool, False): whether the batch also holds the policy state and policy next state of
                each row.

        Returns:
            The :class:`~mushroom_rl.core.batch.Batch` of the parsed rows. A stream of length 1 collapses to the raw
            value.

        Raises:
            AssertionError: if the policy state is requested from a dataset that does not store it.

        """
        if dataset.is_circular:
            dataset = dataset.to_backend(self.backend_name, device=self._device)
            if anchor_idxs is None:
                anchor_idxs = self._backend.arange(0, len(dataset), device=self._device)
            state, next_state, action_history, next_action_history, critic_state, critic_next_state = \
                self._transition_history(dataset.state, dataset.next_state, dataset.action, dataset.last, anchor_idxs,
                                         self._backend, dataset=dataset)
            policy_state, next_policy_state = self._select_policy_state(dataset, anchor_idxs, anchor_idxs) \
                if get_policy_state else (None, None)
            return self._convert_parsed(to, state, dataset.action[anchor_idxs], dataset.reward[anchor_idxs],
                                        next_state, dataset.absorbing[anchor_idxs], dataset.last[anchor_idxs],
                                        action_history, next_action_history, policy_state, next_policy_state,
                                        critic_state, critic_next_state)

        states, actions, reward, next_states, absorbing, last = dataset.parse(
            to=self.backend_name, device=self._device)

        obs, next_obs = self.preprocess(states), self.preprocess(next_states)
        windows = list()
        for name, selection in self._observation_streams().items():
            window, next_window = self._select(obs, selection), self._select(next_obs, selection)
            if name in self._stream_specs:
                window = self._stack_history(name, window, last, attachment=dataset.history_state)
                next_window = self._next_window(name, window, next_window, self._backend)
            windows.append((window, next_window))
        (state, next_state), *critic = windows
        critic_state, critic_next_state = critic[0] if critic and critic[0][0] is not state else (None, None)

        action_history = next_action_history = None
        if self.uses_action:
            action_history = self.build_history('action_history', actions, last, attachment=dataset.history_state)
            next_action_history = self._next_window('action_history', action_history, actions, self._backend)

        if anchor_idxs is not None:
            state, actions, reward = state[anchor_idxs], actions[anchor_idxs], reward[anchor_idxs]
            next_state, absorbing, last = next_state[anchor_idxs], absorbing[anchor_idxs], last[anchor_idxs]
            if self.uses_action:
                action_history, next_action_history = action_history[anchor_idxs], next_action_history[anchor_idxs]
            if critic_state is not None:
                critic_state, critic_next_state = critic_state[anchor_idxs], critic_next_state[anchor_idxs]

        policy_state, next_policy_state = self._select_policy_state(dataset, anchor_idxs, anchor_idxs) \
            if get_policy_state else (None, None)
        return self._convert_parsed(to, state, actions, reward, next_state, absorbing, last, action_history,
                                    next_action_history, policy_state, next_policy_state, critic_state,
                                    critic_next_state)

    def parse_nstep_history(self, dataset, gamma=1., n_steps_return=1, anchor_idxs=None, to=None,
                            get_policy_state=False):
        """
        Parse a dataset into a batch of n-step transitions: :meth:`parse_history` with the discounted n-step reward,
        and the next state, next critic input, next previous-action window, absorbing and last flags of the n-step
        endpoint. Transitions whose return crosses a truncation or runs past the newest stored step are dropped.

        Args:
            dataset (Dataset): the dataset to parse;
            gamma (float, 1.): the discount factor;
            n_steps_return (int, 1): the number of steps of the return;
            anchor_idxs (None): the rows to parse, buffer positions for a circular dataset; by default every row;
            to (str, None): the backend of the returned arrays; by default the agent backend;
            get_policy_state (bool, False): whether the batch also holds the policy state of the first step and the
                policy next state of the endpoint of each transition.

        Returns:
            The :class:`~mushroom_rl.core.batch.Batch` of the kept transitions.

        Raises:
            NotImplementedError: if the agent backend is ``'list'``;
            AssertionError: if the policy state is requested from a dataset that does not store it.

        """
        if self.backend_name == 'list':
            raise NotImplementedError("n-step parsing is not currently supported with the list agent backend.")
        if dataset.is_circular:
            dataset = dataset.to_backend(self.backend_name, device=self._device)
            if anchor_idxs is None:
                anchor_idxs = self._backend.arange(0, len(dataset), device=self._device)
            reduced_reward, anchor, endpoint = self.build_nstep_return(dataset.reward, dataset.absorbing,
                                                                       dataset.last, anchor_idxs, gamma,
                                                                       n_steps_return, dataset=dataset)
            state, next_state, action_history, next_action_history, critic_state, critic_next_state = \
                self._transition_history(dataset.state, dataset.next_state, dataset.action, dataset.last, anchor,
                                         self._backend, next_anchor_idxs=endpoint, dataset=dataset)
            policy_state, next_policy_state = self._select_policy_state(dataset, anchor, endpoint) \
                if get_policy_state else (None, None)
            return self._convert_parsed(to, state, dataset.action[anchor], reduced_reward, next_state,
                                        dataset.absorbing[endpoint], dataset.last[endpoint], action_history,
                                        next_action_history, policy_state, next_policy_state, critic_state,
                                        critic_next_state)

        states, actions, reward, next_states, absorbing, last = dataset.parse(
            to=self.backend_name, device=self._device)
        reduced_reward, anchor, endpoint = self.build_nstep_return(reward, absorbing, last, anchor_idxs, gamma,
                                                                   n_steps_return)

        streams = self._observation_streams()
        stacked = any(name in self._stream_specs for name in streams)
        obs = self.preprocess(states) if stacked else self.preprocess(states[anchor])
        next_obs = self.preprocess(next_states[endpoint])
        windows = list()
        for name, selection in streams.items():
            next_window = self._select(next_obs, selection)
            if name in self._stream_specs:
                stack = self._stack_history(name, self._select(obs, selection), last, attachment=dataset.history_state)
                windows.append((stack[anchor], self._next_window(name, stack[endpoint], next_window, self._backend)))
            else:
                windows.append((self._select(obs[anchor] if stacked else obs, selection), next_window))
        (state, next_state), *critic = windows
        critic_state, critic_next_state = critic[0] if critic and critic[0][0] is not state else (None, None)

        action_history = next_action_history = None
        if self.uses_action:
            windows = self.build_history('action_history', actions, last, attachment=dataset.history_state)
            action_history = windows[anchor]
            next_action_history = self._next_window('action_history', windows[endpoint], actions[endpoint],
                                                    self._backend)

        policy_state, next_policy_state = self._select_policy_state(dataset, anchor, endpoint) if get_policy_state \
            else (None, None)
        return self._convert_parsed(to, state, actions[anchor], reduced_reward, next_state, absorbing[endpoint],
                                    last[endpoint], action_history, next_action_history, policy_state,
                                    next_policy_state, critic_state, critic_next_state)

    def build_history(self, name, buffer, last, anchor_idxs=None, backend=None, attachment=None, dataset=None):
        """
        Build the ``name`` stream window of each anchor from a stored buffer, exactly as :meth:`__call__` builds it
        online. Steps before an episode start, or no longer stored, are zero, unless ``attachment`` holds them.

        Args:
            name (str): the stream;
            buffer: the buffer the stream is read from;
            last: the flags of the final row of every stored segment of the buffer;
            anchor_idxs (None): the row of each window; by default every row;
            backend (ArrayBackend, None): the array backend; by default the agent backend;
            attachment (HistoryState, None): the stream entries preceding the segments that continue rows stored
                elsewhere; ignored with ``anchor_idxs``;
            dataset (Dataset, None): the dataset whose episodes the windows follow with ``anchor_idxs``; by default the
                rows are one stream delimited by ``last``.

        Returns:
            The windows, of shape ``(n_samples, length, *entry_shape)``, oldest entry first, squeezed along ``length``
            when it is 1.

        """
        backend = backend or self._backend
        streams = self._observation_streams()
        if anchor_idxs is not None and name in streams:
            hops, valid = self._get_preceding_observations(buffer, last, anchor_idxs, self._window_span(name), backend,
                                                           dataset)
            return self._window_at(name, streams[name], hops, valid, backend)
        if anchor_idxs is not None:
            return self._build_history_at(name, buffer, last, anchor_idxs, backend, dataset)
        if name in streams:
            buffer = self._select(self.preprocess(buffer), streams[name])

        return self._stack_history(name, buffer, last, backend=backend, attachment=attachment)

    def build_nstep_return(self, reward, absorbing, last, anchor_idxs=None, gamma=1., n_steps_return=1, backend=None,
                           dataset=None):
        """
        Compute the discounted n-step return of a batch of transitions, dropping the ones whose return crosses a
        non-absorbing episode end or runs past the newest stored step. A return ends early at an absorbing step.

        Args:
            reward: the reward of each row;
            absorbing: the absorbing flag of each row;
            last: the flags of the final row of every stored segment;
            anchor_idxs (None): the row of each transition; by default every row;
            gamma (float, 1.): the discount factor;
            n_steps_return (int, 1): the number of steps of the return;
            backend (ArrayBackend, None): the array backend; by default the agent backend;
            dataset (Dataset, None): the dataset whose episodes the return follows with ``anchor_idxs``; by default the
                rows are one stream delimited by ``last``.

        Returns:
            The tuple ``(reward, anchor, endpoint)`` of the kept transitions.

        """
        backend = backend or self._backend
        size = len(reward)
        if anchor_idxs is not None:
            positions, reached = self._walk_forward(dataset, last, anchor_idxs, n_steps_return - 1)
            endpoint = positions[:, -1]
            valid = self._nstep_endpoint_valid(absorbing, last, endpoint, reached)
            acc = reward[anchor_idxs] * gamma ** 0
            for d in range(1, n_steps_return):
                acc = backend.where(reached[:, d], acc + gamma ** d * reward[positions[:, d]], acc)
            return acc[valid], anchor_idxs[valid], endpoint[valid]

        device = backend.get_device(reward)
        offset = backend.zeros(size, dtype=int, device=device)
        valid = backend.ones(size, dtype=bool, device=device)
        active = backend.ones(size, dtype=bool, device=device)
        acc = reward * gamma ** 0
        for t in range(1, n_steps_return):
            tail = size - t
            if tail < 0:
                valid = valid & ~active
                break
            stop = active[:tail + 1] & (last[t - 1:] > 0)
            valid[:tail + 1] = valid[:tail + 1] & ~(stop & (absorbing[t - 1:] <= 0))
            active[:tail + 1] = active[:tail + 1] & ~stop
            valid[tail:] = valid[tail:] & ~active[tail:]
            active[tail:] = False
            body = active[:tail]
            offset[:tail] = backend.where(body, backend.zeros(tail, dtype=int, device=device) + t, offset[:tail])
            acc[:tail] = backend.where(body, acc[:tail] + gamma ** t * reward[t:], acc[:tail])

        anchor_idxs = backend.arange(0, size, device=device)
        endpoint = anchor_idxs + offset
        return acc[valid], anchor_idxs[valid], endpoint[valid]

    def nstep_valid(self, absorbing, last, anchor_idxs=None, n_steps_return=1, backend=None, dataset=None):
        """
        Check which transitions have a well-defined n-step return.

        Args:
            absorbing: the absorbing flag of each row;
            last: the flags of the final row of every stored segment;
            anchor_idxs (None): the row of each transition; by default every row;
            n_steps_return (int, 1): the number of steps of the return;
            backend (ArrayBackend, None): the array backend; by default the agent backend;
            dataset (Dataset, None): the dataset whose episodes the return follows, as in :meth:`build_nstep_return`.

        Returns:
            For each transition, whether its n-step return is well-defined, as kept by :meth:`build_nstep_return`.

        """
        backend = backend or self._backend
        if anchor_idxs is None:
            anchor_idxs = backend.arange(0, len(last), device=backend.get_device(last))
        positions, reached = self._walk_forward(dataset, last, anchor_idxs, n_steps_return - 1)
        return self._nstep_endpoint_valid(absorbing, last, positions[:, -1], reached)

    def history_context(self):
        """
        Returns:
            A :class:`~mushroom_rl.core._impl.history_state.HistoryContext` with, for the observation and the
            previous-action streams, the entries preceding the next step and the entries preceding the most recent
            step, with the environment axis first when the manager is vectorized; ``None`` when no stream has
            entries to carry.

        """
        before_next, before_last = dict(), dict()
        axis = 0 if self._n_envs is None else 1
        for name, window in self._last_windows.items():
            spec = self._stream_specs[name]
            if spec['length'] == 1:
                window = self._backend.expand_dims(window, axis)
            head = (slice(None),) * axis
            if spec['offset'] == 0:
                before_last[name] = window[head + (slice(None, -1),)]
                before_next[name] = window[head + (slice(1, None),)]
            elif name == 'action_history':
                before_last[name] = window
                newest = self._backend.expand_dims(self._last_action, axis)
                before_next[name] = self._backend.concatenate([window[head + (slice(1, None),)], newest], dim=axis)
        return HistoryContext(before_next, before_last) if len(before_next) > 0 else None

    @classmethod
    def default_streams(cls, mdp_info, agent_info, history_length=None, action_history_length=None):
        """
        Build a manager wired up with the default reserved streams read from the MDP and action spaces. When neither
        stream is active the returned manager is the identity context (no stacking).
        The observation stream is registered as ``obs_history`` (offset 0) only when ``history_length`` is greater than
        1; the previous-action stream is registered as ``action_history`` (offset 1) only when ``action_history_length``
        is greater than 0.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            history_length (int, None): number of observations stacked as policy input;
            action_history_length (int, None): number of previous actions stacked as policy input.

        Returns:
            The :class:`HistoryManager` instance.

        """
        return cls(mdp_info, agent_info, streams=cls._default_specs(mdp_info, history_length, action_history_length))

    @property
    def preprocessors(self):
        """
        The preprocessors applied to every observation before it is stacked.

        """
        return self._preprocessors

    @property
    def history_length(self):
        """
        The number of observations stacked as policy input, or 1 when the observation stream is not active.

        """
        return self._stream_specs['obs_history']['length'] if 'obs_history' in self._stream_specs else 1

    @property
    def uses_action(self):
        """
        Whether the previous-action stream is active, i.e. an ``action_history`` window is assembled as policy input.

        """
        return 'action_history' in self._stream_specs

    @property
    def action_history_length(self):
        """
        The number of previous actions stacked as policy input, or 0 when the previous-action stream is not active.

        """
        return self._stream_specs['action_history']['length'] if self.uses_action else 0

    @property
    def max_reach(self):
        """
        The deepest backward reach across all streams, i.e. the maximum of ``offset + length - 1``. A full circular
        buffer reserves this many of its oldest samples so that every window is rebuilt without crossing the write
        head.

        """
        return max((spec['offset'] + spec['length'] - 1
                    for spec in self._stream_specs.values()), default=0)

    def _convert_output(self, to, *arrays):
        """
        Convert the parsed arrays from the agent backend, in which the manager always works, to the one requested by
        the caller. Returns them untouched when no backend is requested or it is the agent's own.

        """
        if to is None or to == self.backend_name:
            return arrays[0] if len(arrays) == 1 else arrays

        return ArrayBackend.get_array_backend(to).convert(*arrays, backend=self._backend)

    def _convert_parsed(self, to, state, action, reward, next_state, absorbing, last, action_history,
                        next_action_history, policy_state, next_policy_state, critic_state, critic_next_state):
        """
        Convert a whole parse result, as in :meth:`_convert_output`, and assemble it into a batch.

        """
        converted = self._convert_output(to, state, action, reward, next_state, absorbing, last)
        if action_history is not None:
            action_history, next_action_history = self._convert_output(to, action_history, next_action_history)
        if policy_state is not None:
            policy_state, next_policy_state = self._convert_output(to, policy_state, next_policy_state)
        if critic_state is not None:
            critic_state, critic_next_state = self._convert_output(to, critic_state, critic_next_state)

        return Batch(*converted, action_history=action_history, next_action_history=next_action_history,
                     policy_state=policy_state, next_policy_state=next_policy_state, critic_state=critic_state,
                     critic_next_state=critic_next_state)

    def _allocate_buffers(self, n_envs):
        self._n_envs = n_envs
        lead = () if n_envs is None else (n_envs,)
        self._buffers = dict()
        for name, spec in self._stream_specs.items():
            if spec['length'] > 1:
                self._buffers[name] = self._backend.zeros(*lead, spec['length'] - 1, *spec['shape'],
                                                          dtype=spec['dtype'], device=self._device)
        self._last_action = None

    def _zero_buffers(self):
        for buffer in self._buffers.values():
            buffer[:] = 0
        self._last_action = None

    def _zero_buffers_vectorized(self, mask):
        mask = self._backend.convert_mask(mask, device=self._device)
        for buffer in self._buffers.values():
            buffer[mask] = 0
        if self._last_action is not None:
            if self.backend_name == 'list':
                self._last_action = [None if reset else action for action, reset in zip(self._last_action, mask)]
            else:
                self._last_action[mask] = 0

    def _stack_history(self, name, buffer, last, backend=None, attachment=None):
        backend = backend or self._backend
        size = buffer.shape[0]
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        device = backend.get_device(buffer)

        out = backend.zeros(size, length, *buffer.shape[1:], dtype=buffer.dtype, device=device)
        active = backend.ones(size, dtype=bool, device=device)
        for t in range(length):
            shift = offset + t
            if shift >= size:
                break
            row_mask = active[shift:].reshape((-1,) + (1,) * (len(buffer.shape) - 1))
            out[shift:, length - 1 - t] = backend.where(row_mask, buffer[:size - shift], out[shift:, length - 1 - t])
            boundary = backend.zeros(size, dtype=bool, device=device)
            boundary[shift:shift + 1] = True
            boundary[shift + 1:] = last[:size - shift - 1] > 0
            active = active & ~boundary

        if offset > 0:
            mask = backend.zeros(size, dtype=bool, device=device)
            for d in range(1, offset + 1):
                mask[d:] = mask[d:] | (last[:size - d] > 0)
            out[mask] = 0
        windows = attachment.windows(name) if attachment is not None else None
        if windows is not None and len(attachment) > 0:
            if len(windows.shape) == len(spec['shape']) + 1:
                windows = backend.expand_dims(windows, 1)
            out = self._attach(out, windows, attachment.positions, last, length, offset, backend, device)
        if length == 1:
            out = out[:, 0]
        return out

    def _transition_history(self, states, next_states, actions, last, anchor_idxs, backend, next_anchor_idxs=None,
                            dataset=None):
        endpoint_idxs = anchor_idxs if next_anchor_idxs is None else next_anchor_idxs
        streams = self._observation_streams()
        reach = max((self._window_span(name) for name in streams if name in self._stream_specs), default=1)

        hops, valid = self._get_preceding_observations(states, last, anchor_idxs, reach, backend, dataset)
        endpoint_hops, endpoint_valid = hops, valid
        if next_anchor_idxs is not None and reach > 1:
            endpoint_hops, endpoint_valid = self._get_preceding_observations(states, last, next_anchor_idxs, reach,
                                                                             backend, dataset)
        next_obs = self.preprocess(next_states[endpoint_idxs])

        windows = list()
        for name, selection in streams.items():
            next_window = self._select(next_obs, selection)
            if name in self._stream_specs:
                window = self._window_at(name, selection, hops, valid, backend)
                endpoint_window = window if next_anchor_idxs is None else \
                    self._window_at(name, selection, endpoint_hops, endpoint_valid, backend)
                windows.append((window, self._next_window(name, endpoint_window, next_window, backend)))
            else:
                windows.append((self._select(hops[0], selection), next_window))
        (state, next_state), *critic = windows
        critic_state, critic_next_state = critic[0] if critic and critic[0][0] is not state else (None, None)

        action_history = next_action_history = None
        if self.uses_action:
            action_history = self._build_history_at('action_history', actions, last, anchor_idxs, backend, dataset)
            if next_anchor_idxs is None:
                endpoint_windows = action_history
            else:
                endpoint_windows = self._build_history_at('action_history', actions, last, next_anchor_idxs, backend,
                                                          dataset)
            next_action_history = self._next_window('action_history', endpoint_windows, actions[endpoint_idxs],
                                                    backend)
        return state, next_state, action_history, next_action_history, critic_state, critic_next_state

    def _build_history_at(self, name, buffer, last, anchor_idxs, backend, dataset):
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        positions, valid = self._walk_back(dataset, last, anchor_idxs, offset + length - 1)
        entries = [buffer[positions[:, hop]] for hop in range(offset, offset + length)]
        return self._stack_indexed(entries, valid[:, offset:], backend)

    def _get_preceding_observations(self, states, last, anchor_idxs, reach, backend, dataset):
        """
        Gather and preprocess the observations of the given rows and of the ``reach - 1`` rows preceding each of them,
        along with whether each preceding row belongs to the same episode.

        """
        if reach == 1:
            return [self.preprocess(states[anchor_idxs])], None
        positions, valid = self._walk_back(dataset, last, anchor_idxs, reach - 1)
        return [self.preprocess(states[positions[:, hop]]) for hop in range(reach)], valid

    def _window_at(self, name, selection, hops, valid, backend):
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        entries = [self._select(hops[hop], selection) for hop in range(offset, offset + length)]
        return self._stack_indexed(entries, valid[:, offset:offset + length], backend)

    def _next_window(self, name, window, entry, backend):
        if self._stream_specs[name]['length'] == 1:
            return entry
        return backend.concatenate([window[:, 1:], entry[:, None]], dim=1)

    def _window_span(self, name):
        spec = self._stream_specs[name]
        return spec['offset'] + spec['length']

    def _check_reserved_streams(self, mdp_info):
        for name, (shape, dtype) in self._reserved_entries(mdp_info).items():
            if name in self._stream_specs:
                spec = self._stream_specs[name]
                dtype = self._backend.to_backend_dtype(dtype)
                assert spec['shape'] == tuple(shape) and spec['dtype'] == dtype, \
                    f"The '{name}' stream holds entries of shape {spec['shape']} and data type {spec['dtype']}, but " \
                    f"its space gives shape {tuple(shape)} and data type {dtype}."

    def _reserved_entries(self, mdp_info):
        """
        The shape and data type of an entry of each reserved stream, as given by the spaces of the MDP.

        """
        return dict(obs_history=(mdp_info.observation_space.shape, mdp_info.observation_space.data_type),
                    action_history=(mdp_info.action_space.shape, mdp_info.action_space.data_type))

    def _observation_streams(self):
        """
        The streams built from the preprocessed observation, mapping each stream name to the selection of the
        observation entries it reads (``None`` for all of them): the policy stream first, then the critic stream when
        there is one. A stream that is not registered is not stacked.

        """
        return {'obs_history': self._actor_selection}

    def _select_policy_state(self, dataset, anchor_idxs, endpoint_idxs):
        assert dataset.is_stateful, "The policy state is requested, but the dataset does not store it."
        policy_state, next_policy_state = self._backend.convert(dataset.policy_state, dataset.policy_next_state,
                                                                device=self._device)
        if anchor_idxs is None:
            return policy_state, next_policy_state
        return policy_state[anchor_idxs], next_policy_state[endpoint_idxs]

    def _stack(self, name, value):
        spec = self._stream_specs[name]
        value = self._missing_value_to_zero(name, value)
        if spec['length'] == 1:
            return value
        return self._append(name, value)

    def _missing_value_to_zero(self, name, value):
        if value is not None:
            return value
        spec = self._stream_specs[name]
        if self._n_envs is not None:
            return self._backend.zeros(self._n_envs, *spec['shape'], dtype=spec['dtype'], device=self._device)
        return self._backend.zeros(*spec['shape'], dtype=spec['dtype'], device=self._device)

    def _append(self, name, value):
        buffer = self._buffers[name]
        if self._n_envs is not None:
            stacked = self._backend.concatenate([buffer, value[:, None]], dim=1)
            buffer[:] = stacked[:, 1:]
        else:
            stacked = self._backend.concatenate([buffer, value[None]], dim=0)
            if isinstance(stacked, list):
                stacked = self._backend.copy(stacked)
            buffer[:] = stacked[1:]

        return stacked

    def _post_load(self, device):
        self._last_windows = dict()

    @staticmethod
    def _select(obs, selection):
        return obs if selection is None else obs[..., selection]

    @staticmethod
    def _stack_indexed(entries, valid, backend):
        """
        Stack the entries of a window, ``entries[t]`` being the one ``t`` steps behind its end and valid where
        ``valid[:, t]`` is; the others are zero.

        """
        length = len(entries)
        n_samples = entries[0].shape[0]
        mask_shape = (n_samples,) + (1,) * (len(entries[0].shape) - 1)
        out = backend.zeros(n_samples, length, *entries[0].shape[1:], dtype=entries[0].dtype,
                            device=backend.get_device(entries[0]))
        for t, entry in enumerate(entries):
            out[:, length - 1 - t] = backend.where(valid[:, t].reshape(mask_shape), entry, out[:, length - 1 - t])
        if length == 1:
            out = out[:, 0]
        return out

    @staticmethod
    def _default_specs(mdp_info, history_length, action_history_length):
        history_length = 1 if history_length is None else history_length
        action_history_length = 0 if action_history_length is None else action_history_length

        streams = dict()
        if history_length > 1:
            streams['obs_history'] = dict(length=history_length, shape=mdp_info.observation_space.shape,
                                          dtype=mdp_info.observation_space.data_type)
        if action_history_length > 0:
            streams['action_history'] = dict(length=action_history_length, shape=mdp_info.action_space.shape,
                                             dtype=mdp_info.action_space.data_type, offset=1)

        return streams

    @staticmethod
    def _attach(out, windows, positions, last, length, offset, backend, device):
        size = len(last)
        rows = backend.arange(0, size, device=device)
        is_start = backend.concatenate([backend.ones(1, dtype=bool, device=device), last[:-1] > 0])
        start_of = backend.where(is_start)[0][backend.cumsum(is_start * 1) - 1]
        entry_of_start = backend.zeros(size, dtype=int, device=device) - 1
        entry_of_start[positions] = backend.arange(0, len(positions), device=device)
        entry = entry_of_start[start_of]
        attached = entry >= 0
        table = backend.concatenate([windows, backend.zeros_like(windows[:1])])
        entry = backend.where(attached, entry, backend.zeros(size, dtype=int, device=device) + len(windows))
        distance = rows - start_of
        reach = windows.shape[1]
        mask_shape = (size,) + (1,) * (len(windows.shape) - 2)
        for t in range(length):
            frame = rows - offset - t
            window_index = distance + length - 1 - t
            use = attached & (frame < start_of) & (window_index >= 0) & (window_index < reach)
            gathered = table[entry, backend.clip(window_index, 0, reach - 1)]
            out[:, length - 1 - t] = backend.where(use.reshape(mask_shape), gathered, out[:, length - 1 - t])
        return out

    @staticmethod
    def _walk_back(dataset, last, anchor_idxs, n_hops):
        if dataset is None:
            return EpisodeLayout.walk_stream_back(last, anchor_idxs, n_hops)
        return dataset.walk_back(anchor_idxs, n_hops, last)

    @staticmethod
    def _walk_forward(dataset, last, anchor_idxs, n_hops):
        if dataset is None:
            return EpisodeLayout.walk_stream_forward(last, anchor_idxs, n_hops)
        return dataset.walk_forward(anchor_idxs, n_hops, last)

    @staticmethod
    def _nstep_endpoint_valid(absorbing, last, endpoint, reached):
        return reached[:, -1] | ((last[endpoint] > 0) & (absorbing[endpoint] > 0))
