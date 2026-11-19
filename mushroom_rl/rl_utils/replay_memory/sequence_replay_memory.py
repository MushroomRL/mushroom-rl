from mushroom_rl.core.batch import SequenceBatch
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
            The :class:`~mushroom_rl.core.batch.SequenceBatch` of the sampled sequences, zero-padded to the truncation
            length, with the previous-action windows of the current and of the next step when that stream is
            active.

        """
        backend = self._dataset.backend
        h = self._history_manager.history_length
        obs_shape = self._mdp_info.observation_space.shape
        obs_dtype = backend.to_backend_dtype(self._mdp_info.observation_space.data_type)
        action_dtype = backend.to_backend_dtype(self._mdp_info.action_space.data_type)

        ds = self._dataset
        size = self.size

        stacked_shape = (h, *obs_shape) if h > 1 else obs_shape
        s = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype,
                          device=self._agent_info.device)
        ss = backend.zeros(n_samples, self._truncation_length, *stacked_shape, dtype=obs_dtype,
                           device=self._agent_info.device)
        a = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                          dtype=action_dtype, device=self._agent_info.device)
        r = backend.zeros(n_samples, device=self._agent_info.device)
        ab = backend.zeros(n_samples, dtype=int, device=self._agent_info.device)
        last = backend.zeros(n_samples, dtype=int, device=self._agent_info.device)
        ps = backend.zeros(n_samples, *self._agent_info.policy_state_shape, device=self._agent_info.device)
        nps = backend.zeros(n_samples, *self._agent_info.policy_state_shape, device=self._agent_info.device)
        lengths = backend.zeros(n_samples, dtype=int,
                                device='cpu' if self._agent_info.backend_name == 'torch' else None)

        stored = backend.arange(0, size, device=self._agent_info.device)
        cut = ds.history_cut(stored, self._history_manager.max_reach)
        candidates = stored[~cut]
        anchors = candidates[backend.randint(0, len(candidates), (n_samples,), device=self._agent_info.device)]

        hops, valid = ds.walk_back(anchors, self._truncation_length - 1)
        counts = (valid & ~cut[hops]).sum(1)
        samples = backend.arange(0, n_samples, device=self._agent_info.device)
        steps = counts[:, None] - 1 - backend.arange(0, self._truncation_length, device=self._agent_info.device)
        positions = hops[samples[:, None], backend.clip(steps, 0, self._truncation_length - 1)]

        sequences = self._history_manager.parse_history(ds, anchor_idxs=positions.reshape(-1), get_policy_state=True)

        def per_sample(x):
            return x.reshape((n_samples, self._truncation_length) + tuple(x.shape[1:]))

        def padded(x, out):
            mask = (steps >= 0).reshape(tuple(steps.shape) + (1,) * (len(x.shape) - 1))
            out[:] = backend.where(mask, per_sample(x), out)
            return out

        padded(sequences.state, s)
        padded(sequences.next_state, ss)
        padded(sequences.action, a)
        ps[:] = per_sample(sequences.policy_state)[:, 0]
        nps[:] = per_sample(sequences.next_policy_state)[:, 0]
        r[:] = per_sample(sequences.reward)[samples, counts - 1]
        ab[:] = per_sample(sequences.absorbing)[samples, counts - 1]
        last[:] = per_sample(sequences.last)[samples, counts - 1]
        lengths[:] = counts

        action_history = next_action_history = None
        if sequences.action_history is not None:
            window_shape = tuple(sequences.action_history.shape[1:])
            window_dtype = sequences.action_history.dtype
            action_history = padded(sequences.action_history,
                                    backend.zeros(n_samples, self._truncation_length, *window_shape,
                                                  dtype=window_dtype, device=self._agent_info.device))
            next_action_history = padded(sequences.next_action_history,
                                         backend.zeros(n_samples, self._truncation_length, *window_shape,
                                                       dtype=window_dtype, device=self._agent_info.device))

        return SequenceBatch(s, a, r, ss, ab, last, action_history=action_history,
                             next_action_history=next_action_history, policy_state=ps, next_policy_state=nps,
                             lengths=lengths)
