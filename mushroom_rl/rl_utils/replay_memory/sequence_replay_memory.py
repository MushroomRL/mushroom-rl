from mushroom_rl.rl_utils.replay_memory.replay_memory import ReplayMemory


class SequenceReplayMemory(ReplayMemory):
    """
    This class extend the base replay memory to allow sampling sequences of a certain length. This is useful for
    training recurrent agents or agents operating on a window of states etc.

    The temporal length of the sampled sequences (``truncation_length``) and the history-stacking length of each
    timestep (``history_length``, carried by the injected :class:`HistoryManager`) are orthogonal and compose: with a
    ``history_length`` greater than 1 each timestep of the sequence is itself a stacked window, so the sampled states
    have shape ``(n_samples, truncation_length, history_length, *obs_shape)``, collapsing to
    ``(n_samples, truncation_length, *obs_shape)`` when no stacking is used.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, truncation_length, history_manager=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            truncation_length (int): truncation length to be sampled;
            history_manager (HistoryManager, None): the manager used by the agent to assemble the stacked observation
                online, reused to rebuild the same stacked observation for every timestep of a sequence.

        """
        self._truncation_length = truncation_length

        super().__init__(mdp_info, agent_info, initial_size, max_size,
                         history_manager=history_manager, store_policy_state=True)

        self._add_save_attr(
            _truncation_length='primitive'
        )

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        backend = self._dataset.array_backend
        h = self._history_length
        max_size = self._max_size
        obs_shape = self._mdp_info.observation_space.shape
        obs_dtype = self._mdp_info.observation_space.data_type

        # work in chronological offsets from the oldest sample (the write head when full), mapping to buffer positions
        # with a modulo so that a wrapped buffer is read in trajectory order rather than in raw index order
        start = self._idx if self._full else 0
        size = self.size
        # a valid anchor must be far enough from the write head that no window (observation or previous action) crosses
        # it, and the sequence is stopped there too, so every timestep keeps a valid window (its length is truncated)
        min_offset = self._max_reach if self._full else 0

        stacked_shape = (h, *obs_shape) if h > 1 else obs_shape
        s = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype)
        ss = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype)
        a = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                          dtype=self._mdp_info.action_space.data_type)
        r = backend.zeros(n_samples, 1)
        ab = backend.zeros(n_samples, 1, dtype=int)
        last = backend.zeros(n_samples, dtype=int)
        ps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        nps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        pa = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                           dtype=self._mdp_info.action_space.data_type)
        lengths = list()

        for num, c_anchor in enumerate(backend.randint(min_offset, size, (n_samples,))):
            c_anchor = int(c_anchor)
            c_begin = max(c_anchor - self._truncation_length + 1, min_offset)

            # stop the sequence at the most recent episode boundary inside the window
            window = backend.arange(c_begin, c_anchor)
            if len(window) > 0:
                boundary = backend.where(self._dataset.last[(start + window) % max_size] > 0)
                if len(boundary[0]) > 0:
                    c_begin = c_begin + int(boundary[0][-1]) + 1

            length = c_anchor - c_begin + 1
            positions = (start + backend.arange(c_begin, c_anchor + 1)) % max_size
            anchor = (start + c_anchor) % max_size

            # rebuild every per-timestep window of the sequence (stacked observation and, when active, the previous
            # action) through the history manager, exactly as the base replay memory does for a single transition
            if self._history_manager is not None:
                state_seq, next_state_seq, extra = self._history_manager.build_transition_windows_circular_buffer(
                    self._dataset.state, self._dataset.next_state, self._dataset.action,
                    self._dataset.last, positions, len(self._dataset), self._full, self._max_size)
            else:
                state_seq, next_state_seq, extra = (self._dataset.state[positions],
                                                    self._dataset.next_state[positions], dict())

            s[num, :length] = state_seq
            ss[num, :length] = next_state_seq
            a[num, :length] = self._dataset.action[positions]
            ps[num, :length] = self._dataset.policy_state[positions]
            nps[num, :length] = self._dataset.policy_next_state[positions]
            if 'action_history' in extra:
                pa[num, :length] = extra['action_history']
            r[num] = self._dataset.reward[anchor]
            ab[num] = self._dataset.absorbing[anchor]
            last[num] = self._dataset.last[anchor]

            lengths.append(length)

        if self._dataset.is_stateful:
            return s, a, r, ss, ab, last, ps, nps, pa, lengths
        else:
            return s, a, r, ss, ab, last, pa, lengths
