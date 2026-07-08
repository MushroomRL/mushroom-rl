from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend


class HistoryManager(MushroomObject):
    """
    Object in charge of assembling the per-timestep context fed to the policy, i.e. the stacked window of the most
    recent entries of one or more streams.

    The context is a deterministic function of the observed trajectory, hence it is always reconstructable from the
    stored transitions and is not part of the (latent) policy state. The manager holds an ordered set of named streams,
    each with its own stacking length and an ``offset`` telling how many steps behind the current one its window ends
    (0 for the observation, 1 for the previous action). The reserved ``obs`` stream is delivered in-band as the policy
    ``state``; every other stream is returned as a keyword argument. At least one stream must be active.

    The manager works entirely in the agent backend and is a pure stacker: online it stacks whatever it is fed, and the
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
        self._vectorized = False

        self._add_save_attr(
            _agent_backend='primitive',
            _stream_specs='primitive',
            _buffers='none',
            _vectorized='primitive'
        )

        if obs_history_length > 1:
            self.add_stream('obs', obs_history_length, mdp_info.observation_space.shape,
                            mdp_info.observation_space.data_type)
        for name, spec in (extra_buffers or dict()).items():
            self.add_stream(name, **spec)

        assert self._stream_specs, "HistoryManager requires at least one active stream."

    def add_stream(self, name, length, shape, dtype, offset=0, **options):
        """
        Register a named stream to be stacked by the manager. Streams are usually declared through the constructor, but
        this method allows building a manager and adding buffers programmatically.

        Args:
            name (str): the name under which the stream's window is returned by :meth:`__call__` (``obs`` is reserved
                for the in-band observation stream);
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
        Build a manager from the MDP and agent information, returning ``None`` when no stream is active. The
        previous-action stream, when requested, is registered as the reserved ``action_history`` named stream with
        ``offset`` 1, deriving its shape and data type from the action space.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            history_length (int, None): number of observations stacked as policy input;
            action_history_length (int, None): number of previous actions stacked as policy input.

        Returns:
            The :class:`HistoryManager` instance, or ``None`` if neither the observation nor the action stream is
            active.

        """
        history_length = 1 if history_length is None else history_length
        action_history_length = 0 if action_history_length is None else action_history_length

        if history_length <= 1 and action_history_length <= 0:
            return None

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
        return self._stream_specs['obs']['length'] if 'obs' in self._stream_specs else 1

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
    def exogenous_streams(self):
        """
        Names of the active streams that are not derived from a stored transition column, i.e. neither the observation
        stream nor the reserved ``action_history`` stream. These streams are fed online only and cannot be rebuilt
        offline from a replay memory, so a manager carrying them is rejected when injected into one.

        """
        return set(self._stream_specs) - {'obs', 'action_history'}

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
        else:
            start_mask = self._agent_backend.convert(start_mask)
            for buffer in self._buffers.values():
                buffer[start_mask] = 0

    def __call__(self, state=None, **extra):
        """
        Append the current entries to the buffers and return the per-timestep context split for the policy call: the
        observation input to be passed positionally as ``state`` and a dictionary of the additional conditioning
        streams to be forwarded as keyword arguments. The reserved ``obs`` stream, when active, replaces ``state`` with
        its stacked window; otherwise the raw ``state`` is passed through unchanged. Each other stream is forwarded
        under its own name (the previous-action stream under ``action_history``).

        Args:
            state: the current observation, already in the agent backend, consumed by the observation stream;
            **extra: the current value of each non-observation stream, keyed by its name and already in the agent
                backend; a stream whose value is not provided is zero-padded for that step.

        Returns:
            A tuple ``(state, policy_kwargs)`` ready to be used as ``policy.draw_action(state, **policy_kwargs)``. Each
            window has shape ``(length, *shape)`` (single-environment) or ``(n_envs, length, *shape)`` (vectorized),
            squeezed along the ``length`` axis when the stream length is 1.

        """
        windows = dict()
        for name in self._stream_specs:
            value = state if name == 'obs' else extra.get(name)
            windows[name] = self._stack(name, value)
        if 'obs' in self._stream_specs:
            return windows.pop('obs'), windows
        return state, windows

    def _shift(self, array, k):
        """
        Shift ``array`` down by ``k`` rows along its first axis, filling the first ``k`` rows with zeros, i.e. row
        ``i`` of the result is row ``i - k`` of ``array`` (zero when ``i < k``). Used to walk the whole buffer at once
        without materializing an index array.

        """
        shifted = self._agent_backend.zeros(*array.shape, dtype=array.dtype)
        if k < array.shape[0]:
            shifted[k:] = array[:array.shape[0] - k]
        return shifted

    def build_history(self, name, buffer, last, anchor_idxs=None):
        """
        Rebuild the ``name`` stream window offline for a batch of anchor indices, reading from a regular (non-circular)
        buffer such as an in-memory dataset. Each window is built by walking backwards from its anchor up to the stream
        length, stopping at the start of the buffer or at an episode boundary and zero-padding the missing older
        entries, which reproduces exactly the window assembled online by :meth:`__call__`.

        The stream ``offset`` ends the window ``offset`` steps before the anchor (e.g. 1 for the previous action). When
        the anchor is within ``offset`` steps of its trajectory start the window would reach into the previous episode,
        so it is zeroed instead, matching the online reset at every episode start.

        When ``anchor_idxs`` is ``None`` the window of every timestep of the buffer is rebuilt, and the walk advances by
        plain row shifts instead of gathering, so no index array is allocated.

        Args:
            name (str): the stream to rebuild, providing its length and offset;
            buffer: the buffer to read from;
            last: the ``last`` flags of the buffer, used to stop at episode boundaries;
            anchor_idxs (None): buffer indices of the current step of each window; when ``None`` every timestep of the
                buffer is an anchor.

        Returns:
            An array of shape ``(n_samples, length, *entry_shape)`` (squeezed along ``length`` when it is 1), with older
            entries at lower indices.

        """
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        backend = self._agent_backend
        size = buffer.shape[0]

        if anchor_idxs is None:
            mask_shape = (size,) + (1,) * (len(buffer.shape) - 1)
            out = backend.zeros(size, length, *buffer.shape[1:], dtype=buffer.dtype)
            active = backend.ones(size, dtype=bool)
            for t in range(length):
                shift = offset + t
                in_range = backend.ones(size, dtype=bool)
                in_range[:shift] = False
                valid = active & in_range
                out[:, length - 1 - t] = backend.where(valid.reshape(mask_shape), self._shift(buffer, shift),
                                                       out[:, length - 1 - t])
                boundary = self._shift(last, shift + 1) > 0     # row i: last[i - shift - 1] > 0
                if shift < size:
                    boundary[shift] = True                      # row i == shift: the walk reached the anchor's start
                active = valid & ~boundary

            if offset > 0:
                mask = backend.zeros(size, dtype=bool)
                for d in range(1, offset + 1):
                    mask = mask | (self._shift(last, d) > 0)
                out[mask] = 0
            if length == 1:
                out = out[:, 0]
            return out

        n_samples = len(anchor_idxs)
        mask_shape = (n_samples,) + (1,) * (len(buffer.shape) - 1)
        out = backend.zeros(n_samples, length, *buffer.shape[1:], dtype=buffer.dtype)

        walk_anchors = anchor_idxs - offset
        active = backend.ones(n_samples, dtype=bool)
        for t in range(length):
            pos = walk_anchors - t
            valid = active & (pos >= 0) & (pos < size)
            gather = buffer[backend.clip(pos, 0, size - 1)]
            out[:, length - 1 - t] = backend.where(valid.reshape(mask_shape), gather, out[:, length - 1 - t])
            boundary = (pos == 0) | ((pos > 0) & (last[backend.clip(pos - 1, 0, size - 1)] > 0))
            active = valid & ~boundary

        if offset > 0:
            mask = (anchor_idxs >= 1) & (last[backend.clip(anchor_idxs - 1, 0, size - 1)] > 0)
            for d in range(2, offset + 1):
                mask = mask | ((anchor_idxs >= d) & (last[backend.clip(anchor_idxs - d, 0, size - 1)] > 0))
            out[mask] = 0
        if length == 1:
            out = out[:, 0]
        return out

    def build_history_circular_buffer(self, name, buffer, last, anchor_idxs, size, full, max_size):
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
            max_size (int): the maximum size of the circular buffer.

        Returns:
            An array of shape ``(n_samples, length, *entry_shape)`` (squeezed along ``length`` when it is 1), with older
            entries at lower indices.

        """
        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        backend = self._agent_backend
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

    def build_transition_windows(self, states, next_states, actions, last):
        """
        Rebuild every automatically-handled window of a full regular buffer (e.g. an in-memory dataset), one window
        per timestep, from its state, next-state and action columns. See
        :meth:`build_transition_windows_circular_buffer` for the circular replay-buffer variant.

        Args:
            states: the state column of the buffer;
            next_states: the next-state column of the buffer;
            actions: the action column of the buffer;
            last: the ``last`` flags of the buffer, used to stop at episode boundaries.

        Returns:
            A tuple ``(state, next_state, extra)`` with the state and next-state windows and a dictionary of the
            remaining stream windows keyed by name (see the online policy keyword arguments of :meth:`__call__`).

        """
        return self._transition_windows(states, next_states, actions, last)

    def build_transition_windows_circular_buffer(self, states, next_states, actions, last, anchor_idxs, size, full,
                                                 max_size):
        """
        Rebuild every automatically-handled window for a batch of anchors of a circular replay buffer, from its state,
        next-state and action columns. Same result as :meth:`build_transition_windows`, reading the circular buffer at
        ``anchor_idxs`` (see :meth:`build_history_circular_buffer` for the ``size``/``full``/``max_size`` arguments).

        Args:
            states: the state column of the buffer;
            next_states: the next-state column of the buffer;
            actions: the action column of the buffer;
            last: the ``last`` flags of the buffer, used to stop at episode boundaries;
            anchor_idxs: buffer indices of the current step of each transition;
            size (int): the number of valid entries currently stored in the buffer;
            full (bool): whether the circular buffer has wrapped around;
            max_size (int): the maximum size of the circular buffer.

        Returns:
            A tuple ``(state, next_state, extra)`` with the state and next-state windows and a dictionary of the
            remaining stream windows keyed by name.

        """
        return self._transition_windows(states, next_states, actions, last, anchor_idxs, size, full, max_size)

    def _rebuild_stream(self, name, buffer, last, anchor_idxs, size, full, max_size):
        # dispatch one stream's reconstruction to the regular builder (whole buffer) or the circular one (anchors)
        if anchor_idxs is None:
            return self.build_history(name, buffer, last)
        return self.build_history_circular_buffer(name, buffer, last, anchor_idxs, size, full, max_size)

    def _transition_windows(self, states, next_states, actions, last, anchor_idxs=None, size=None, full=False,
                            max_size=None):
        # shared assembly for the two transition builders: the observation stream is stacked from the state and
        # next-state columns (or gathered as is when inactive), every other stream is read from the column it is
        # bound to and returned in the extra dictionary; a ``None`` anchor reconstructs a whole regular buffer
        columns = dict(action_history=actions)  # column each non-observation stream is bound to

        if 'obs' in self._stream_specs:
            state = self._rebuild_stream('obs', states, last, anchor_idxs, size, full, max_size)
            next_state = self._rebuild_stream('obs', next_states, last, anchor_idxs, size, full, max_size)
        elif anchor_idxs is None:
            state, next_state = states, next_states
        else:
            state, next_state = states[anchor_idxs], next_states[anchor_idxs]

        extra = {name: self._rebuild_stream(name, columns[name], last, anchor_idxs, size, full, max_size)
                 for name in self._stream_specs if name != 'obs'}
        return state, next_state, extra

    def _stack(self, name, value):
        spec = self._stream_specs[name]
        length = spec['length']
        stacked, self._buffers[name] = self._append(self._buffers[name], value, spec['shape'], spec['dtype'])
        if length == 1:
            stacked = stacked[:, 0] if self._vectorized else stacked[0]
        return stacked

    def _append(self, buffer, value, shape, dtype):
        if value is None:
            if self._vectorized:
                value = self._agent_backend.zeros(buffer.shape[0], *shape, dtype=dtype)
            else:
                value = self._agent_backend.zeros(*shape, dtype=dtype)

        if self._vectorized:
            stacked = self._agent_backend.concatenate([buffer, value[:, None]], dim=1)
            new_buffer = self._agent_backend.copy(stacked[:, 1:])
        else:
            stacked = self._agent_backend.concatenate([buffer, value[None]], dim=0)
            new_buffer = self._agent_backend.copy(stacked[1:])

        return stacked, new_buffer
