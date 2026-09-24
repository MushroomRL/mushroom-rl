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
    def __init__(self, mdp_info, agent_info, initial_size, max_size, truncation_length, history_manager=None,
                 return_extra=False):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            truncation_length (int): truncation length to be sampled;
            history_manager (HistoryManager, None): the manager used by the agent to assemble the stacked observation
                online, reused to rebuild the same stacked observation for every timestep of a sequence;
            return_extra (bool, False): whether :meth:`get` appends, as a trailing element, the ``extra_data``
                dictionary of the history windows not delivered in-band in the state, keyed as the online policy
                keyword arguments. When ``False`` these streams are not returned.

        """
        self._truncation_length = truncation_length

        super().__init__(mdp_info, agent_info, initial_size, max_size,
                         history_manager=history_manager, store_policy_state=True, return_extra=return_extra)

        self._add_save_attr(
            _truncation_length='primitive'
        )

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples. When ``return_extra`` is set, the ``extra_data`` dictionary assembled
            by the history manager, padded to the truncation length, is appended as the trailing element.

        """
        backend = self._dataset.array_backend
        h = self._history_manager.history_length
        max_size = self._max_size
        obs_shape = self._mdp_info.observation_space.shape
        obs_dtype = backend.to_backend_dtype(self._mdp_info.observation_space.data_type)
        action_dtype = backend.to_backend_dtype(self._mdp_info.action_space.data_type)

        ds = self._dataset
        start = ds.write_head if ds.full else 0
        size = self.size
        min_offset = self._history_manager.max_reach if ds.full else 0

        stacked_shape = (h, *obs_shape) if h > 1 else obs_shape
        s = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype,
                          device=self._agent_info.device)
        ss = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype,
                           device=self._agent_info.device)
        a = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                          dtype=action_dtype, device=self._agent_info.device)
        r = backend.zeros(n_samples, 1, device=self._agent_info.device)
        ab = backend.zeros(n_samples, 1, dtype=int, device=self._agent_info.device)
        last = backend.zeros(n_samples, dtype=int, device=self._agent_info.device)
        ps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape,
                           device=self._agent_info.device)
        nps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape,
                            device=self._agent_info.device)

        extra_buffers = dict()
        lengths = list()

        if ds.links is None:
            anchors = backend.randint(min_offset, size, (n_samples,), device=self._agent_info.device)
        else:
            cut = self._history_cut(backend.arange(0, size, device=self._agent_info.device))
            candidates = backend.arange(0, size, device=self._agent_info.device)[~cut]
            anchors = candidates[backend.randint(0, len(candidates), (n_samples,), device=self._agent_info.device)]

        for num, c_anchor in enumerate(anchors):
            c_anchor = int(c_anchor)
            if ds.links is None:
                c_begin = max(c_anchor - self._truncation_length + 1, min_offset)

                window = backend.arange(c_begin, c_anchor, device=self._agent_info.device)
                if len(window) > 0:
                    boundary = backend.where(ds.last[(start + window) % max_size] > 0)
                    if len(boundary[0]) > 0:
                        c_begin = c_begin + int(boundary[0][-1]) + 1

                length = c_anchor - c_begin + 1
                positions = (start + backend.arange(c_begin, c_anchor + 1, device=self._agent_info.device)) % max_size
            else:
                positions = self._sequence_positions(c_anchor, cut)
                length = len(positions)

            state_seq, action_seq, reward_seq, next_state_seq, absorbing_seq, last_seq, extra = \
                self._history_manager.parse_history_circular_buffer(
                    ds, positions, len(ds), ds.full, self._max_size, links=ds.links, write_head=ds.write_head)

            s[num, :length] = state_seq
            ss[num, :length] = next_state_seq
            a[num, :length] = action_seq
            ps[num, :length] = self._dataset.policy_state[positions]
            nps[num, :length] = self._dataset.policy_next_state[positions]
            if self._return_extra:
                for name, value in extra.items():
                    if name not in extra_buffers:
                        extra_buffers[name] = backend.zeros(n_samples, self._truncation_length, *value.shape[1:],
                                                            dtype=value.dtype, device=self._agent_info.device)
                    extra_buffers[name][num, :length] = value
            r[num] = reward_seq[-1]
            ab[num] = absorbing_seq[-1]
            last[num] = last_seq[-1]

            lengths.append(length)

        out = [s, a, r, ss, ab, last, ps, nps, lengths]
        if self._return_extra:
            out.append(extra_buffers)
        return tuple(out)

    def _sequence_positions(self, anchor, cut):
        """
        Args:
            anchor (int): the buffer position of the final step of the sequence;
            cut: for every stored position, whether its history window reaches a step that is not stored anymore.

        Returns:
            The buffer positions of the steps of the episode of ``anchor`` ending at it, oldest first, at most
            ``truncation_length`` of them, all stored and with a stored history window.

        """
        ds = self._dataset
        prev = ds.links[0]
        steps = [anchor]
        while len(steps) < self._truncation_length:
            distance = int(prev[steps[-1]])
            age = (steps[-1] - ds.write_head) % self._max_size if ds.full else steps[-1]
            previous = (steps[-1] - distance) % self._max_size
            if distance == 0 or distance > age or bool(cut[previous]):
                break
            steps.append(previous)
        return ds.array_backend.as_array(steps[::-1], device=self._agent_info.device)
