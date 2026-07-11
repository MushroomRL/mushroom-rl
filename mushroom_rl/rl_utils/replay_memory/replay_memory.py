from mushroom_rl.core import DatasetInfo, Dataset, MushroomObject
from mushroom_rl.core.history_manager import HistoryManager


class ReplayMemory(MushroomObject):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size,
                 history_manager=None, n_steps_return=1, store_policy_state=False, return_extra=False):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            history_manager (HistoryManager, None): the manager used by the agent to assemble the stacked observation,
                reused offline so that the stacked observation matches the one built online;
            n_steps_return (int, 1): number of steps used for the n-step return;
            store_policy_state (bool, False): whether the policy internal state is stored in the replay memory. When
                ``False``, no policy-state buffer is allocated and the policy state of the added datasets is dropped;
                a stateless algorithm should leave it ``False`` even if its policy is stateful;
            return_extra (bool, False): whether :meth:`get` appends, as a trailing element, the ``extra_data``
                dictionary of the history windows not delivered in-band in the state, keyed as the online policy
                keyword arguments. When ``False`` these streams are not returned.

        """
        assert agent_info.backend in ["numpy", "torch"], \
            f"{agent_info.backend} backend currently not supported in the replay memory class."

        self._initial_size = initial_size
        self._max_size = max_size
        self._history_manager = history_manager if history_manager is not None \
            else HistoryManager.from_infos(mdp_info, agent_info)
        self._n_steps_return = n_steps_return
        self._store_policy_state = store_policy_state
        self._return_extra = return_extra
        self._mdp_info = mdp_info
        self._agent_info = agent_info

        self._idx = 0
        self._full = False
        self._dataset = None
        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _history_manager='mushroom',
            _n_steps_return='primitive',
            _store_policy_state='primitive',
            _return_extra='primitive',
            _mdp_info='mushroom',
            _agent_info='mushroom',
            _idx='primitive!',
            _full='primitive!',
            _dataset='mushroom!',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): dataset class elements to add to the replay memory.

        """
        assert not self._dataset.is_stateful or dataset.is_stateful, \
            "The replay memory is configured to store the policy state, but the dataset does not provide it."

        dataset = dataset.to_backend(self._agent_info.backend)
        self._write_to_buffer(dataset)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        idxs = self._sample_idxs(n_samples)
        return tuple(self._assemble_batch(idxs))

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info,
                                                              self._store_policy_state)
        self._dataset = Dataset(dataset_info, n_steps=self._max_size)

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that allows it to be used.

        """
        return self.size >= self._initial_size

    def _assemble_batch(self, idxs):
        """
        Read the transitions at the given buffer indices and assemble the batch.
        When a history is used the stacked observation windows are rebuilt from the buffer. The policy states
        are appended when stored, followed by the ``extra_data`` dictionary of the out-of-band history windows when
        ``return_extra`` is set.

        Args:
            idxs: the buffer indices of the transitions to read.

        Returns:
            The list of arrays forming the sampled batch.

        """
        ds = self._dataset
        state, action, reward, next_state, absorbing, last, extra = \
            self._history_manager.parse_nstep_history_circular_buffer(
                ds, idxs, self._mdp_info.gamma, self._n_steps_return, len(ds), self._full, self._max_size, self._idx)
        endpoint = extra.pop('endpoint')

        policy_state = [ds.policy_state[idxs], ds.policy_next_state[endpoint]] if ds.is_stateful else []

        out = [state, action, reward, next_state, absorbing, last, *policy_state]

        if self._return_extra:
            out.append(extra)

        return out

    def _sample_idxs(self, n_samples):
        """
        Sample buffer indices to read, drawing uniformly among the anchors that can be sampled, i.e. those whose
        stacked observation window can be rebuilt and whose n-step return can be completed (see :meth:`_compute_mask`).

        Args:
            n_samples (int): the number of indices to sample.

        Returns:
            The sampled buffer indices.

        """
        backend = self._dataset.array_backend
        size = len(self._dataset)
        if self._history_manager.max_reach == 0 and self._n_steps_return == 1:
            return backend.randint(0, size, (n_samples,))
        idxs = backend.arange(0, size)
        valid = idxs[~self._compute_mask(idxs)]
        return valid[backend.randint(0, len(valid), (n_samples,))]

    def _affected_window(self, positions):
        """
        The buffer positions whose sampling mask can change after a batch was written at ``positions``: the newly
        written anchors, their forward n-step window (the ``n-1`` anchors ending in the new batch) and the backward
        history reserve that trails the moved write head. Every other entry keeps its mask.

        Args:
            positions: the buffer positions where the last batch was written.

        Returns:
            The affected buffer positions, or ``None`` when no masking is in use.

        """
        if self._history_manager.max_reach == 0 and self._n_steps_return == 1:
            return None

        backend = self._dataset.array_backend
        size = len(self._dataset)
        history_reserve = self._history_manager.max_reach if self._full else 0
        window_length = len(positions) + (self._n_steps_return - 1) + history_reserve
        raw = (positions[0] - (self._n_steps_return - 1)) + backend.arange(0, window_length)
        if self._full:
            return raw % self._max_size
        return raw[(raw >= 0) & (raw < size)]

    def _compute_mask(self, anchor_idxs):
        """
        Compute the sampling mask for a batch of anchors: True where the anchor cannot be sampled because its n-step
        window would cross a truncation or the write head, or because its backward history window would cross the write
        head of a full buffer.

        Args:
            anchor_idxs: buffer positions of the anchors to evaluate.

        Returns:
            The boolean mask (True = excluded from sampling) for the given anchors.

        """
        backend = self._dataset.array_backend
        mask = backend.zeros(len(anchor_idxs), dtype=bool)
        if self._n_steps_return > 1:
            *_, valid = self._history_manager.parse_nstep_return_circular_buffer(
                self._dataset.reward, self._dataset.absorbing, self._dataset.last, anchor_idxs, self._mdp_info.gamma,
                self._n_steps_return, len(self._dataset), self._full, self._max_size, self._idx)
            mask = mask | ~valid
        if self._history_manager.max_reach > 0 and self._full:
            mask = mask | ((anchor_idxs - self._idx) % self._max_size < self._history_manager.max_reach)
        return mask

    def _write_to_buffer(self, dataset):
        """
        Write transitions from a dataset into the circular buffer.

        Uses ``append_batch`` while the buffer still has capacity, then switches to
        direct slice assignment once the buffer is full, wrapping around as needed.

        Args:
            dataset (Dataset): transitions to write.

        Returns:
            The buffer positions (indices into the circular buffer) where the
            transitions were written.

        """
        n = len(dataset)
        backend = self._dataset.array_backend
        positions = (backend.arange(0, n) + self._idx) % self._max_size

        if not self._full:
            remaining = self._max_size - len(self._dataset)
            if n <= remaining:
                self._dataset.append_batch(dataset)
                self._idx += n
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0
                return positions

            self._dataset.append_batch(dataset[:remaining])
            self._full = True
            self._idx = 0
            dataset = dataset[remaining:]
            n -= remaining

        end = self._idx + n
        if end <= self._max_size:
            self._dataset.state[self._idx:end] = dataset.state
            self._dataset.action[self._idx:end] = dataset.action
            self._dataset.reward[self._idx:end] = dataset.reward
            self._dataset.next_state[self._idx:end] = dataset.next_state
            self._dataset.absorbing[self._idx:end] = dataset.absorbing
            self._dataset.last[self._idx:end] = dataset.last
            if self._dataset.is_stateful:
                self._dataset.policy_state[self._idx:end] = dataset.policy_state
                self._dataset.policy_next_state[self._idx:end] = dataset.policy_next_state
            self._idx = end % self._max_size
        else:
            first = self._max_size - self._idx
            rest = n - first
            self._dataset.state[self._idx:] = dataset.state[:first]
            self._dataset.state[:rest] = dataset.state[first:]
            self._dataset.action[self._idx:] = dataset.action[:first]
            self._dataset.action[:rest] = dataset.action[first:]
            self._dataset.reward[self._idx:] = dataset.reward[:first]
            self._dataset.reward[:rest] = dataset.reward[first:]
            self._dataset.next_state[self._idx:] = dataset.next_state[:first]
            self._dataset.next_state[:rest] = dataset.next_state[first:]
            self._dataset.absorbing[self._idx:] = dataset.absorbing[:first]
            self._dataset.absorbing[:rest] = dataset.absorbing[first:]
            self._dataset.last[self._idx:] = dataset.last[:first]
            self._dataset.last[:rest] = dataset.last[first:]
            if self._dataset.is_stateful:
                self._dataset.policy_state[self._idx:] = dataset.policy_state[:first]
                self._dataset.policy_state[:rest] = dataset.policy_state[first:]
                self._dataset.policy_next_state[self._idx:] = dataset.policy_next_state[:first]
                self._dataset.policy_next_state[:rest] = dataset.policy_next_state[first:]
            self._idx = rest

        return positions

    def _post_load(self):
        if self._full is None:
            self.reset()
