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
    is fed as a keyword argument to :meth:`__call__` and returned under its own name. At least one stream must be
    active.

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

        assert self._stream_specs, "HistoryManager requires at least one active stream."

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
        window at the next step (its ``offset`` 1). Called by the agent after every ``draw_action``; a no-op when the
        previous-action stream is not active.

        Args:
            action: the action just drawn, already in the agent backend.

        """
        if self.uses_action:
            self._last_action = action

    def build_history(self, name, buffer, last, anchor_idxs=None):
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
                buffer is an anchor.

        Returns:
            An array of shape ``(n_samples, length, *entry_shape)`` (squeezed along ``length`` when it is 1), with older
            entries at lower indices.

        """
        size = buffer.shape[0]
        if anchor_idxs is not None:
            return self.build_history_circular_buffer(name, buffer, last, anchor_idxs, size, full=False, max_size=size)

        spec = self._stream_specs[name]
        length, offset = spec['length'], spec['offset']
        backend = self._agent_backend

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
        if 'obs_history' in self._stream_specs:
            state = self.build_history('obs_history', states, last)
            next_state = self.build_history('obs_history', next_states, last)
        else:
            state, next_state = states, next_states

        extra = dict()
        if self.uses_action:
            extra['action_history'] = self.build_history('action_history', actions, last)
        return state, next_state, extra

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
        if 'obs_history' in self._stream_specs:
            state = self.build_history_circular_buffer('obs_history', states, last, anchor_idxs, size, full, max_size)
            next_state = self.build_history_circular_buffer('obs_history', next_states, last, anchor_idxs, size, full,
                                                            max_size)
        else:
            state, next_state = states[anchor_idxs], next_states[anchor_idxs]

        extra = dict()
        if self.uses_action:
            extra['action_history'] = self.build_history_circular_buffer('action_history', actions, last, anchor_idxs,
                                                                         size, full, max_size)
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
