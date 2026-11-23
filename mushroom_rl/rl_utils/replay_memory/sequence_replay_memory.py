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

        ds = self._dataset
        size = self.size

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

        s, ss, *critic = [self._pad_sequences(array, steps) for array in sequences.state_arrays]
        a, *history = [self._pad_sequences(array, steps) for array in sequences.action_arrays]
        critic_state, critic_next_state = critic or (None, None)
        action_history, next_action_history = history or (None, None)
        final = samples * self._truncation_length + counts - 1
        ps[:] = sequences.policy_state[::self._truncation_length]
        nps[:] = sequences.next_policy_state[::self._truncation_length]
        r[:] = sequences.reward[final]
        ab[:] = sequences.absorbing[final]
        last[:] = sequences.last[final]
        lengths[:] = counts

        return SequenceBatch(s, a, r, ss, ab, last, action_history=action_history,
                             next_action_history=next_action_history, policy_state=ps, next_policy_state=nps,
                             critic_state=critic_state, critic_next_state=critic_next_state, lengths=lengths)

    @staticmethod
    def _pad_sequences(x, steps):
        """
        Reshape the per-step array into sequences, with the entries of the padding steps (negative ``steps``) set to
        zero.

        """
        x = x.reshape(tuple(steps.shape) + tuple(x.shape[1:]))
        x[steps < 0] = 0
        return x
