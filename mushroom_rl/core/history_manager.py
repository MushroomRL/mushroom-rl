from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend


class HistoryManager(MushroomObject):
    """
    Object in charge of assembling the per-timestep context fed to the policy, i.e. the stacked window of the most
    recent entries of one or more streams.

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
    same stacking rule is exposed offline through :meth:`build_history` (regular buffer) and
    :meth:`build_history_circular_buffer` (circular replay buffer), so the window built while interacting with the
    environment and the one rebuilt from a stored buffer are guaranteed to match. Each per-step window is
    ``(length, *shape)``, squeezed to ``(*shape)`` when ``length`` is 1.

    Each stream is described by a specification dictionary with the keys ``length``, ``shape`` and ``dtype`` plus any
    number of options; the only option acted upon by this class is ``offset`` (default 0). Subclasses may store and
    honor richer options (strides, structured lookbacks) without changing the base machinery.

    """
    def __init__(self, mdp_info, agent_info, obs_history_length=1, extra_buffers=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP, providing the shape and data type of the observation stream;
            agent_info (AgentInfo): information about the agent, providing the array backend in which the manager keeps
                its buffers and returns the stacked windows;
            obs_history_length (int, 1): number of observations stacked as policy input; the observation stream is
                active only when this is greater than 1;
            extra_buffers (dict, None): the named streams stacked on top of the observation one, given as a mapping
                ``name -> spec``, where each ``spec`` is a dictionary with the keys ``length``, ``shape`` and ``dtype``
                and, optionally, ``offset`` (default 0), the number of steps behind the current one at which the window
                ends. Each spec is forwarded to :meth:`add_stream`. Each stream's window is returned under ``name`` in
                the keyword-argument output of :meth:`__call__`. The previous-action stream is one such stream,
                conventionally named ``action_history`` with ``offset`` 1, wired up by the agent through
                :meth:`from_infos`.

        """
        self._agent_backend = ArrayBackend.get_array_backend(agent_info.backend)
        self._stream_specs = dict()
        self._buffers = None
        self._last_action = None
        self._vectorized = False

        self._add_save_attr(
            _agent_backend='primitive',
            _stream_specs='primitive',
            _buffers='none',
            _last_action='none',
            _vectorized='primitive'
        )

        if obs_history_length > 1:
            self.add_stream('obs_history', obs_history_length, mdp_info.observation_space.shape,
                            mdp_info.observation_space.data_type)
        for name, spec in (extra_buffers or dict()).items():
            self.add_stream(name, **spec)

    def add_stream(self, name, length, shape, dtype, offset=0, **options):
        """
        Register a named stream to be stacked by the manager. Streams are usually declared through the constructor, but
        this method allows building a manager and adding buffers programmatically.

        Args:
            name (str): the name under which the stream's window is returned by :meth:`__call__` (``obs_history`` is
                reserved for the in-band observation stream);
            length (int): number of entries stacked in the stream's window;
            shape (tuple): shape of a single entry of the stream;
            dtype: data type of the stream, converted to the agent backend;
            offset (int, 0): number of steps behind the current one at which the window ends;
            **options: additional per-stream options stored in the specification; ignored by the base class.

        """
        self._stream_specs[name] = dict(length=length, shape=tuple(shape),
                                        dtype=self._agent_backend.to_backend_dtype(dtype), offset=offset, **options)

    @classmethod
    def from_infos(cls, mdp_info, agent_info, history_length=None, action_history_length=None):
        """
        Build a manager from the MDP and agent information. When neither stream is active the returned manager is the
        identity context (no stacking), so callers can always hold a manager rather than a ``None``. The
        previous-action stream, when requested, is registered as the reserved ``action_history`` named stream with
        ``offset`` 1, deriving its shape and data type from the action space.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            history_length (int, None): number of observations stacked as policy input;
            action_history_length (int, None): number of previous actions stacked as policy input.

        Returns:
            The :class:`HistoryManager` instance.

        """
        history_length = 1 if history_length is None else history_length
        action_history_length = 0 if action_history_length is None else action_history_length

        extra_buffers = dict()
        if action_history_length > 0:
            extra_buffers['action_history'] = dict(length=action_history_length, shape=mdp_info.action_space.shape,
                                                   dtype=mdp_info.action_space.data_type, offset=1)

        return cls(mdp_info, agent_info, obs_history_length=history_length, extra_buffers=extra_buffers)

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

    def reset(self):
        """
        Reset the buffers at the beginning of a single-environment episode.

        """
        self._vectorized = False
        self._last_action = None
        self._buffers = {name: self._agent_backend.zeros(spec['length'] - 1, *spec['shape'], dtype=spec['dtype'])
                         for name, spec in self._stream_specs.items()}

    def reset_vectorized(self, start_mask):
        """
        Reset the buffers for the environments selected by ``start_mask``, leaving the others untouched.

        Args:
            start_mask: boolean mask selecting the environments that are starting a new episode.

        """
        self._vectorized = True
        n_envs = len(start_mask)
        existing_buffers = self._buffers.values() if self._buffers else []
        reference_buffer = next(iter(existing_buffers), None)

        if reference_buffer is None or reference_buffer.shape[0] != n_envs:
            self._buffers = {name: self._agent_backend.zeros(n_envs, spec['length'] - 1, *spec['shape'],
                                                             dtype=spec['dtype'])
                             for name, spec in self._stream_specs.items()}
            self._last_action = None
        else:
            start_mask = self._agent_backend.convert(start_mask)
            for buffer in self._buffers.values():
                buffer[start_mask] = 0
            if self.uses_action and self._last_action is not None:
                self._last_action[start_mask] = 0

    def __call__(self, state=None, **extra):
        """
        Append the current entries to the buffers and return the per-timestep context split for the policy call: the
        observation input to be passed positionally as ``state`` and a dictionary of the additional conditioning
        streams to be forwarded as keyword arguments. The reserved ``obs_history`` stream, when active, replaces
        ``state`` with its stacked window; otherwise the raw ``state`` is passed through unchanged. The
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
        windows = dict()
        for name in self._stream_specs:
            if name == 'obs_history':
                value = state
            elif name == 'action_history':
                value = self._last_action
            else:
                value = extra.get(name)
            windows[name] = self._stack(name, value)
        if 'obs_history' in self._stream_specs:
            return windows.pop('obs_history'), windows
        return state, windows

    def record_action(self, action):
        """
        Record the action just drawn by the agent so that it becomes the most recent entry of the ``action_history``
        window at the next step (its ``offset`` 1). Called by the agent after every ``draw_action``. The last action
        is always kept (it is cheap and available for custom logging), even when the previous-action stream is not
        active and it is not stacked into any window.

        Args:
            action: the action just drawn, already in the agent backend.

        """
        self._last_action = action

    def build_history(self, name, buffer, last, anchor_idxs=None, backend=None):
        """
        Rebuild the ``name`` stream window offline for a batch of anchor indices, reading from a regular (non-circular)
        buffer such as an in-memory dataset. Each window is built by walking backwards from its anchor up to the stream
        length, stopping at the start of the buffer or at an episode boundary and zero-padding the missing older
        entries, which reproduces exactly the window assembled online by :meth:`__call__`.

        The stream ``offset`` ends the window ``offset`` steps before the anchor (e.g. 1 for the previous action). When
        the anchor is within ``offset`` steps of its trajectory start the window would reach into the previous episode,
        so it is zeroed instead, matching the online reset at every episode start.

        When ``anchor_idxs`` is ``None`` every timestep of the buffer is an anchor (i.e. ``0..size-1``) and each
        backward step is a uniform row shift done by slicing; otherwise the explicit anchors are gathered through
        :meth:`build_history_circular_buffer` (a regular buffer is the never-wrapped ``full=False`` case).

        Args:
            name (str): the stream to rebuild, providing its length and offset;
            buffer: the buffer to read from;
            last: the ``last`` flags of the buffer, used to stop at episode boundaries;
            anchor_idxs (None): buffer indices of the current step of each window; when ``None`` every timestep of the
                buffer is an anchor;
            backend (ArrayBackend, None): the backend used to allocate the window; when ``None`` the agent backend is
                used.

        Returns:
            An array of shape ``(n_samples, length, *entry_shape)`` (squeezed along ``length`` when it is 1), with older
            entries at lower indices.

        """
        backend = backend or self._agent_backend
        size = buffer.shape[0]
        if anchor_idxs is not None:
            return self.build_history_circular_buffer(name, buffer, last, anchor_idxs, size, full=False, max_size=size,
                                                      backend=backend)

        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']

        out = backend.zeros(size, length, *buffer.shape[1:], dtype=buffer.dtype)
        active = backend.ones(size, dtype=bool)
        for t in range(length):
            shift = offset + t
            if shift >= size:
                break
            row_mask = active[shift:].reshape((-1,) + (1,) * (len(buffer.shape) - 1))
            out[shift:, length - 1 - t] = backend.where(row_mask, buffer[:size - shift], out[shift:, length - 1 - t])
            boundary = backend.zeros(size, dtype=bool)
            boundary[shift] = True
            boundary[shift + 1:] = last[:size - shift - 1] > 0
            active = active & ~boundary

        if offset > 0:
            mask = backend.zeros(size, dtype=bool)
            for d in range(1, offset + 1):
                mask[d:] = mask[d:] | (last[:size - d] > 0)
            out[mask] = 0
        if length == 1:
            out = out[:, 0]
        return out

    def build_history_circular_buffer(self, name, buffer, last, anchor_idxs, size, full, max_size, backend=None):
        """
        Same as :meth:`build_history`, but reading from a circular replay buffer: positions are taken modulo the buffer
        size, the walk stops both at episode boundaries and at the buffer limits (the start of a not-yet-wrapped buffer,
        which is the first stored episode start, and the write head of a full one, which the anchors are assumed to stay
        clear of, see :attr:`max_reach`).

        Args:
            name (str): the stream to rebuild, providing its length and offset;
            buffer: the circular buffer to read from (e.g. the state or action column of a replay memory);
            last: the ``last`` flags of the buffer, used to stop at episode boundaries;
            anchor_idxs: buffer indices of the current step of each window;
            size (int): the number of valid entries currently stored in the buffer;
            full (bool): whether the circular buffer has wrapped around;
            max_size (int): the maximum size of the circular buffer;
            backend (ArrayBackend, None): the backend used to allocate the window; when ``None`` the agent backend is
                used.

        Returns:
            An array of shape ``(n_samples, length, *entry_shape)`` (squeezed along ``length`` when it is 1), with older
            entries at lower indices.

        """
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        backend = backend or self._agent_backend
        n_samples = len(anchor_idxs)
        mask_shape = (n_samples,) + (1,) * (len(buffer.shape) - 1)
        out = backend.zeros(n_samples, length, *buffer.shape[1:], dtype=buffer.dtype)

        walk_anchors = anchor_idxs - offset
        active = backend.ones(n_samples, dtype=bool)
        for t in range(length):
            pos = walk_anchors - t
            if full:
                valid = active
                gather_idx, prev_idx = pos % max_size, (pos - 1) % max_size
            else:
                valid = active & (pos >= 0) & (pos < size)
                gather_idx, prev_idx = backend.clip(pos, 0, size - 1), backend.clip(pos - 1, 0, size - 1)
            out[:, length - 1 - t] = backend.where(valid.reshape(mask_shape), buffer[gather_idx],
                                                   out[:, length - 1 - t])
            boundary = last[prev_idx] > 0
            if not full:
                boundary = (pos == 0) | boundary
            active = valid & ~boundary

        if offset > 0:
            edge = (anchor_idxs - 1) % max_size if full else backend.clip(anchor_idxs - 1, 0, size - 1)
            mask = last[edge] > 0
            for d in range(2, offset + 1):
                edge = (anchor_idxs - d) % max_size if full else backend.clip(anchor_idxs - d, 0, size - 1)
                mask = mask | (last[edge] > 0)
            out[mask] = 0
        if length == 1:
            out = out[:, 0]
        return out

    def parse_history(self, dataset, to=None):
        """
        Parse a dataset into its arrays, the analog of :meth:`Dataset.parse` with the history stacking rules applied:
        the state and next-state windows replace the raw observations with the stack of their most recent entries, i.e.
        the temporal context fed to the policy, and the window of every other active stream (e.g. the previous actions)
        is returned aside.

        Args:
            dataset (Dataset): the dataset to parse;
            to (str, None): the backend of the returned arrays; when ``None`` the agent backend is used.

        Returns:
            The tuple ``(state, action, reward, next_state, absorbing, last, extra)``, as :meth:`Dataset.parse` plus a
            dictionary mapping every other active stream name to its window; ``state`` and ``next_state`` carry the
            stacked windows. A stream stacking a single entry collapses to the raw value.

        """
        backend = ArrayBackend.get_array_backend(to) if to else self._agent_backend
        dataset = dataset.to_backend(backend.get_backend_name())
        states, actions, reward = dataset.state, dataset.action, dataset.reward
        next_states, absorbing, last = dataset.next_state, dataset.absorbing, dataset.last

        if 'obs_history' in self._stream_specs:
            state = self.build_history('obs_history', states, last, backend=backend)
            next_state = self.build_history('obs_history', next_states, last, backend=backend)
        else:
            state, next_state = states, next_states

        extra = dict()
        if self.uses_action:
            extra['action_history'] = self.build_history('action_history', actions, last, backend=backend)
        return state, actions, reward, next_state, absorbing, last, extra

    def parse_history_circular_buffer(self, dataset, anchor_idxs, size, full, max_size, to=None):
        """
        Parse the transitions at ``anchor_idxs`` of a dataset stored in a circular replay buffer, as in
        :meth:`parse_history`. A circular buffer overwrites its oldest entry once full, so a window is read modulo the
        capacity and is never stitched across the write head.

        Args:
            dataset (Dataset): the dataset to parse;
            anchor_idxs: the buffer position of each transition to parse;
            size (int): the number of entries currently stored;
            full (bool): whether the buffer has wrapped around;
            max_size (int): the buffer capacity;
            to (str, None): the backend of the returned arrays; when ``None`` the agent backend is used.

        Returns:
            The tuple ``(state, action, reward, next_state, absorbing, last, extra)``, as in :meth:`parse_history`.

        """
        backend = ArrayBackend.get_array_backend(to) if to else self._agent_backend
        dataset = dataset.to_backend(backend.get_backend_name())
        state, next_state, extra = self._transition_history(dataset.state, dataset.next_state, dataset.action,
                                                            dataset.last, anchor_idxs, anchor_idxs, size, full,
                                                            max_size, backend)
        return (state, dataset.action[anchor_idxs], dataset.reward[anchor_idxs], next_state,
                dataset.absorbing[anchor_idxs], dataset.last[anchor_idxs], extra)

    def parse_nstep_return(self, reward, absorbing, last, anchor_idxs=None, gamma=1., n_steps_return=1, backend=None):
        """
        Compute the n-step return of a batch of transitions. The n-step return of a transition is the discounted sum of
        the rewards collected over the next ``n_steps_return`` steps; its bootstrap target is the transition
        ``n_steps_return`` steps ahead (the endpoint), or the terminal transition if the episode ends earlier.

        Args:
            reward: the reward of each transition;
            absorbing: the absorbing flag of each transition;
            last: the episode-boundary flags of the buffer;
            anchor_idxs (None): the buffer position of each transition; when ``None`` every stored transition is used;
            gamma (float, 1.): the discount factor;
            n_steps_return (int, 1): the number of steps summed in the return;
            backend (ArrayBackend, None): the backend used to allocate the arrays; when ``None`` the agent backend is
                used.

        Returns:
            The tuple ``(endpoint, reduced_reward, valid)``: the index of the bootstrap transition, the discounted
            n-step reward, and whether the n-step return is well defined (``False`` when the window would cross a
            truncated episode). ``endpoint`` and ``reduced_reward`` are meaningful only where ``valid``.

        """
        backend = backend or self._agent_backend
        size = len(reward)
        if anchor_idxs is None:
            anchor_idxs = backend.arange(0, size)
        return self.parse_nstep_return_circular_buffer(reward, absorbing, last, anchor_idxs, gamma, n_steps_return,
                                                       size, full=False, max_size=size, write_head=size,
                                                       backend=backend)

    def parse_nstep_return_circular_buffer(self, reward, absorbing, last, anchor_idxs, gamma, n_steps_return, size,
                                           full, max_size, write_head, backend=None):
        """
        Compute the n-step return of a batch of transitions stored in a circular replay buffer, as in
        :meth:`parse_nstep_return`. An absorbing terminal ends the return early and is the bootstrap target; a
        non-absorbing truncation, or a window reaching past the newest stored transition (the write head), makes the
        transition invalid.

        Args:
            reward: the reward column of the buffer;
            absorbing: the absorbing column of the buffer;
            last: the episode-boundary flags of the buffer;
            anchor_idxs: the buffer position of each transition;
            gamma (float): the discount factor;
            n_steps_return (int): the number of steps summed in the return;
            size (int): the number of entries currently stored;
            full (bool): whether the buffer has wrapped around;
            max_size (int): the buffer capacity;
            write_head (int): the next write position; the newest stored transition is the one before it;
            backend (ArrayBackend, None): the backend used to allocate the arrays; when ``None`` the agent backend is
                used.

        Returns:
            The tuple ``(endpoint, reduced_reward, valid)``, as in :meth:`parse_nstep_return`.

        """
        backend = backend or self._agent_backend
        n_samples = len(anchor_idxs)
        max_offset = (write_head - 1 - anchor_idxs) % max_size if full else size - 1 - anchor_idxs
        offset = backend.zeros(n_samples, dtype=int)
        acc = reward[anchor_idxs] * gamma ** 0
        active = backend.ones(n_samples, dtype=bool)
        valid = backend.ones(n_samples, dtype=bool)
        for t in range(1, n_steps_return):
            prev = (anchor_idxs + (t - 1)) % max_size if full else backend.clip(anchor_idxs + (t - 1), 0, size - 1)
            stop = active & (last[prev] > 0)
            valid = valid & ~(stop & (absorbing[prev] <= 0))
            active = active & ~stop
            ran_off = active & (t > max_offset)
            valid = valid & ~ran_off
            active = active & ~ran_off
            cur = (anchor_idxs + t) % max_size if full else backend.clip(anchor_idxs + t, 0, size - 1)
            offset = backend.where(active, backend.zeros(n_samples, dtype=int) + t, offset)
            acc = backend.where(active, acc + gamma ** t * reward[cur], acc)
        endpoint = (anchor_idxs + offset) % max_size if full else anchor_idxs + offset
        return endpoint, acc, valid

    def parse_nstep_history(self, dataset, gamma=1., n_steps_return=1, anchor_idxs=None, to=None):
        """
        Parse a dataset into its n-step arrays, i.e. :meth:`parse_history` with the n-step return folded in: the reward
        becomes the discounted n-step reward and the next-state window, the absorbing and the last flags belong to the
        n-ahead endpoint, while the state and previous-action windows belong to the current step.

        Args:
            dataset (Dataset): the dataset to parse;
            gamma (float, 1.): the discount factor;
            n_steps_return (int, 1): the number of steps summed in the return;
            anchor_idxs (None): the buffer position of each transition; when ``None`` every stored transition is used;
            to (str, None): the backend of the returned arrays; when ``None`` the agent backend is used.

        Returns:
            The tuple ``(state, action, reward, next_state, absorbing, last, extra)``, as in
            :meth:`parse_nstep_history_circular_buffer`.

        """
        size = len(dataset)
        if anchor_idxs is None:
            backend = ArrayBackend.get_array_backend(to) if to else self._agent_backend
            anchor_idxs = backend.arange(0, size)
        return self.parse_nstep_history_circular_buffer(dataset, anchor_idxs, gamma, n_steps_return, size,
                                                        full=False, max_size=size, write_head=size, to=to)

    def parse_nstep_history_circular_buffer(self, dataset, anchor_idxs, gamma, n_steps_return, size, full, max_size,
                                            write_head, to=None):
        """
        Parse the transitions at ``anchor_idxs`` of a dataset stored in a circular replay buffer into their n-step
        arrays, as in :meth:`parse_nstep_history`.

        Args:
            dataset (Dataset): the dataset to parse;
            anchor_idxs: the buffer position of each transition;
            gamma (float): the discount factor;
            n_steps_return (int): the number of steps summed in the return;
            size (int): the number of entries currently stored;
            full (bool): whether the buffer has wrapped around;
            max_size (int): the buffer capacity;
            write_head (int): the next write position of the buffer;
            to (str, None): the backend of the returned arrays; when ``None`` the agent backend is used.

        Returns:
            The tuple ``(state, action, reward, next_state, absorbing, last, extra)``, as :meth:`parse_history` but with
            the discounted n-step reward and the endpoint next-state, absorbing and last. The endpoint indices are
            provided under ``extra['endpoint']``.

        """
        backend = ArrayBackend.get_array_backend(to) if to else self._agent_backend
        dataset = dataset.to_backend(backend.get_backend_name())
        endpoint, reduced_reward, _ = self.parse_nstep_return_circular_buffer(
            dataset.reward, dataset.absorbing, dataset.last, anchor_idxs, gamma, n_steps_return, size, full, max_size,
            write_head, backend=backend)
        state, next_state, extra = self._transition_history(dataset.state, dataset.next_state, dataset.action,
                                                            dataset.last, anchor_idxs, endpoint, size, full, max_size,
                                                            backend)
        extra['endpoint'] = endpoint
        return (state, dataset.action[anchor_idxs], reduced_reward, next_state,
                dataset.absorbing[endpoint], dataset.last[endpoint], extra)

    def _transition_history(self, states, next_states, actions, last, anchor_idxs, next_anchor_idxs, size, full,
                            max_size, backend):
        if 'obs_history' in self._stream_specs:
            state = self.build_history_circular_buffer('obs_history', states, last, anchor_idxs, size, full, max_size,
                                                       backend=backend)
            next_state = self.build_history_circular_buffer('obs_history', next_states, last, next_anchor_idxs, size,
                                                            full, max_size, backend=backend)
        else:
            state, next_state = states[anchor_idxs], next_states[next_anchor_idxs]

        extra = dict()
        if self.uses_action:
            extra['action_history'] = self.build_history_circular_buffer('action_history', actions, last, anchor_idxs,
                                                                         size, full, max_size, backend=backend)
        return state, next_state, extra

    def _stack(self, name, value):
        spec = self._stream_specs[name]
        value = self._missing_value_to_zero(name, value)
        stacked = self._append(name, value)
        if spec['length'] == 1:
            stacked = stacked[:, 0] if self._vectorized else stacked[0]
        return stacked

    def _missing_value_to_zero(self, name, value):
        if value is not None:
            return value
        spec = self._stream_specs[name]
        if self._vectorized:
            return self._agent_backend.zeros(self._buffers[name].shape[0], *spec['shape'], dtype=spec['dtype'])
        return self._agent_backend.zeros(*spec['shape'], dtype=spec['dtype'])

    def _append(self, name, value):
        buffer = self._buffers[name]
        if self._vectorized:
            stacked = self._agent_backend.concatenate([buffer, value[:, None]], dim=1)
            buffer[:] = stacked[:, 1:]
        else:
            stacked = self._agent_backend.concatenate([buffer, value[None]], dim=0)
            buffer[:] = stacked[1:]

        return stacked
