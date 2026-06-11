from mushroom_rl.core import DatasetInfo, Dataset, Serializable


class ReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size,
                 history_length=1, n_steps_return=1):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            history_length (int, 1): number of consecutive observations returned per sample;
            n_steps_return (int, 1): number of steps used for the n-step return.

        """
        assert agent_info.backend in ["numpy", "torch"], \
            f"{agent_info.backend} backend currently not supported in the replay memory class."

        self._initial_size = initial_size
        self._max_size = max_size
        self._history_length = history_length
        self._n_steps_return = n_steps_return
        self._mdp_info = mdp_info
        self._agent_info = agent_info

        self._idx = 0
        self._full = False
        self._dataset = None
        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _history_length='primitive',
            _n_steps_return='primitive',
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
        assert self._dataset.is_stateful == dataset.is_stateful

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
        idxs = self._dataset.array_backend.randint(0, len(self._dataset), (n_samples,))

        if self._history_length > 1:
            state_out, nstate_out = self._get_with_history(idxs)
            batch = self._dataset[idxs]
            _, action, reward, _, absorbing, last = batch.parse()
            return state_out, action, reward, nstate_out, absorbing, last

        dataset_batch = self._dataset[idxs]

        if self._dataset.is_stateful:
            return *dataset_batch.parse(), *dataset_batch.parse_policy_state()
        else:
            return dataset_batch.parse()

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info)
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
            if dataset.is_stateful:
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
            if dataset.is_stateful:
                self._dataset.policy_state[self._idx:] = dataset.policy_state[:first]
                self._dataset.policy_state[:rest] = dataset.policy_state[first:]
                self._dataset.policy_next_state[self._idx:] = dataset.policy_next_state[:first]
                self._dataset.policy_next_state[:rest] = dataset.policy_next_state[first:]
            self._idx = rest

        return positions

    def _get_with_history(self, idxs):
        """
        Retrieve state and next-state observations with history stacking for the given indices.

        Args:
            idxs (array): buffer indices to retrieve.

        Returns:
            A tuple ``(state_out, next_state_out)`` where each array has shape 
            ``(n_samples, history_length, *obs_shape)``.

        """
        state_out = self._build_history(idxs, self._dataset.state)
        nstate_out = self._build_history(idxs, self._dataset.next_state)
        return state_out, nstate_out

    def _build_history(self, anchor_idxs, buffer):
        """
        Build a stacked observation history for each anchor index by walking backwards
        through the circular buffer up to ``history_length`` steps, stopping at episode boundaries.

        Args:
            anchor_idxs (array): buffer indices to use as the most-recent observation;
            buffer (array): the observation buffer to read from (state or next_state).

        Returns:
            An array of shape ``(n_samples, history_length, *obs_shape)`` with older observations at lower channel 
            indices and the anchor observation at index ``history_length - 1``.

        """
        dataset = self._dataset
        is_full = self._full
        h = self._history_length
        obs_shape = buffer.shape[1:]
        obs_dtype = buffer.dtype
        n_samples = len(anchor_idxs)
        out = self._dataset.array_backend.zeros(n_samples, h, *obs_shape, dtype=obs_dtype)

        for k, anchor in enumerate(anchor_idxs):
            for t in range(h):
                pos = anchor - t
                if not is_full and (pos < 0 or pos >= len(dataset)):
                    break
                buf = pos % self._max_size
                out[k, h - 1 - t] = buffer[buf]
                if t < h - 1:
                    prev = (pos - 1) % self._max_size
                    if (not is_full and pos == 0) or dataset.last[prev]:
                        break

        return out

    def _post_load(self):
        if self._full is None:
            self.reset()