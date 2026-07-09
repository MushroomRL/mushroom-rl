from mushroom_rl.core import DatasetInfo, Dataset, MushroomObject


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
        self._history_manager = history_manager
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

    @property
    def _history_length(self):
        return self._history_manager.history_length if self._history_manager is not None else 1

    @property
    def _max_reach(self):
        """
        The deepest backward reach across all history streams, i.e. how many of the oldest samples of a full buffer are
        excluded from sampling so that no window is rebuilt across the write head. It is 0 when no history is used.

        """
        return self._history_manager.max_reach if self._history_manager is not None else 0

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): dataset class elements to add to the replay memory.

        """
        assert not self._dataset.is_stateful or dataset.is_stateful, \
            "The replay memory is configured to store the policy state, but the dataset does not provide it."

        if self._n_steps_return > 1:
            state, action, reward, next_state, absorbing, last = dataset.parse(to=self._agent_info.backend)
            policy_state, policy_next_state = (dataset.parse_policy_state(to=self._agent_info.backend)
                                               if self._dataset.is_stateful else (None, None))
            result = self._compute_n_step_return(state, action, reward, next_state, absorbing, last,
                                                 policy_state, policy_next_state)
            if result is None:
                return
            dataset, _ = result
        else:
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
        batch = self._dataset[idxs]
        state, action, reward, next_state, absorbing, last = batch.parse()

        extra = dict()
        if self._history_manager is not None:
            state, next_state, extra = self._history_manager.build_transition_windows_circular_buffer(
                self._dataset.state, self._dataset.next_state, self._dataset.action,
                self._dataset.last, idxs, len(self._dataset), self._full, self._max_size)

        out = [state, action, reward, next_state, absorbing, last]

        if self._dataset.is_stateful:
            out += list(batch.parse_policy_state())

        if self._return_extra:
            out.append(extra)

        return tuple(out)

    def _sample_idxs(self, n_samples):
        """
        Sample buffer indices to read, excluding anchors for which a valid stacked observation cannot be rebuilt.

        When the buffer is full and a history is used, the ``max_reach`` oldest samples are skipped: their earlier
        entries have already been overwritten, so walking back from them would cross the write head and stitch together
        entries from unrelated trajectory segments.

        Args:
            n_samples (int): the number of indices to sample.

        Returns:
            The sampled buffer indices.

        """
        backend = self._dataset.array_backend
        if self._max_reach > 0 and self._full:
            offsets = backend.randint(self._max_reach, self._max_size, (n_samples,))
            return (self._idx + offsets) % self._max_size
        return backend.randint(0, len(self._dataset), (n_samples,))

    def _crosses_write_head(self, idx):
        """
        Whether a window anchored at buffer position ``idx`` would cross the write head of a full buffer, i.e. ``idx``
        is one of the ``max_reach`` oldest samples whose earlier entries were overwritten.

        """
        return self._max_reach > 0 and self._full and \
            (idx - self._idx) % self._max_size < self._max_reach

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info, self._store_policy_state)
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

    def _compute_n_step_return(self, state, action, reward, next_state, absorbing, last,
                               policy_state=None, policy_next_state=None, priority=None):
        """
        Compute the n-step discounted return for each valid transition in a batch and
        return them as a Dataset together with the filtered priorities.

        A transition at index i is valid if the n-step lookahead from i does not cross a
        non-absorbing episode boundary. Transitions that would require bootstrapping past
        a premature ``last`` flag (non-absorbing) are skipped.

        Args:
            state (array): batch of states;
            action (array): batch of actions;
            reward (array): batch of immediate rewards;
            next_state (array): batch of next states;
            absorbing (array): batch of absorbing flags;
            last (array): batch of last-step flags;
            policy_state (array, None): batch of policy internal states;
            policy_next_state (array, None): batch of next policy internal states;
            priority (array, None): per-transition priorities; filtered to valid indices when provided.

        Returns:
            A tuple ``(Dataset, priority)`` with the valid transitions and their filtered priorities,
            or ``None`` if no valid transitions exist.

        """
        backend = self._dataset.array_backend
        max_valid = len(state) - self._n_steps_return + 1
        if max_valid <= 0:
            return None
        valid_i = backend.zeros(max_valid, dtype=int)
        valid_ij = backend.zeros(max_valid, dtype=int)
        valid_reward = backend.zeros(max_valid, dtype=reward.dtype)
        count = 0
        i = 0
        while i < max_valid:
            j = 0
            skip = False
            acc_reward = reward[i]
            while j < self._n_steps_return - 1:
                if last[i + j]:
                    if not absorbing[i + j]:
                        skip = True
                        i += j + 1
                    break
                j += 1
                acc_reward = acc_reward + self._mdp_info.gamma ** j * reward[i + j]
            if not skip:
                valid_i[count] = i
                valid_ij[count] = i + j
                valid_reward[count] = acc_reward
                count += 1
                i += 1

        if count == 0:
            return None

        valid_i = valid_i[:count]
        valid_ij = valid_ij[:count]

        ps = policy_state[valid_i] if policy_state is not None else None
        pns = policy_next_state[valid_ij] if policy_next_state is not None else None
        p = priority[valid_i] if priority is not None else None

        dataset = Dataset.from_array(state[valid_i], action[valid_i], valid_reward[:count],
                                     next_state[valid_ij], absorbing[valid_ij], last[valid_ij],
                                     policy_state=ps, policy_next_state=pns,
                                     backend=self._agent_info.backend)
        return dataset, p

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
