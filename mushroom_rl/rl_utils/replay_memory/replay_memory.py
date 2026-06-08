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
            history_length (int, 1): number of consecutive frames returned per sample;
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

        state, action, reward, next_state, absorbing, last = dataset.parse(to=self._agent_info.backend)

        if self._dataset.is_stateful:
            policy_state, policy_next_state = dataset.parse_policy_state(to=self._agent_info.backend)
        else:
            policy_state, policy_next_state = None, None

        if self._n_steps_return > 1:
            result = self._compute_n_step_return(state, action, reward, next_state, absorbing, last,
                                                 policy_state, policy_next_state)
            if result is None:
                return
            state, action, reward, next_state, absorbing, last, policy_state, policy_next_state, _ = result

        self._write_to_buffer(state, action, reward, next_state, absorbing, last,
                              policy_state, policy_next_state)

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
        return self.size >= self._initial_size

    def _compute_n_step_return(self, state, action, reward, next_state, absorbing, last,
                               policy_state=None, policy_next_state=None, priority=None):
        backend = self._dataset.array_backend
        max_valid = len(state) - self._n_steps_return + 1
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

        return (state[valid_i], action[valid_i], valid_reward[:count],
                next_state[valid_ij], absorbing[valid_ij], last[valid_ij],
                ps, pns, p)

    def _write_to_buffer(self, state, action, reward, next_state, absorbing, last,
                         policy_state, policy_next_state):
        n = len(state)
        backend = self._dataset.array_backend
        positions = (backend.arange(0, n) + self._idx) % self._max_size

        if not self._full:
            remaining = self._max_size - len(self._dataset)
            if n <= remaining:
                self._dataset.append_batch(state, action, reward, next_state, absorbing, last,
                                           policy_state, policy_next_state)
                self._idx += n
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0
                return positions

            self._dataset.append_batch(
                state[:remaining], action[:remaining], reward[:remaining],
                next_state[:remaining], absorbing[:remaining], last[:remaining],
                policy_state[:remaining] if policy_state is not None else None,
                policy_next_state[:remaining] if policy_next_state is not None else None
            )
            self._full = True
            self._idx = 0
            state, action, reward = state[remaining:], action[remaining:], reward[remaining:]
            next_state, absorbing, last = next_state[remaining:], absorbing[remaining:], last[remaining:]
            if policy_state is not None:
                policy_state = policy_state[remaining:]
                policy_next_state = policy_next_state[remaining:]
            n -= remaining

        end = self._idx + n
        if end <= self._max_size:
            self._dataset.state[self._idx:end] = state
            self._dataset.action[self._idx:end] = action
            self._dataset.reward[self._idx:end] = reward
            self._dataset.next_state[self._idx:end] = next_state
            self._dataset.absorbing[self._idx:end] = absorbing
            self._dataset.last[self._idx:end] = last
            if policy_state is not None:
                self._dataset.policy_state[self._idx:end] = policy_state
                self._dataset.policy_next_state[self._idx:end] = policy_next_state
            self._idx = end % self._max_size
        else:
            first = self._max_size - self._idx
            rest = n - first
            self._dataset.state[self._idx:] = state[:first]
            self._dataset.state[:rest] = state[first:]
            self._dataset.action[self._idx:] = action[:first]
            self._dataset.action[:rest] = action[first:]
            self._dataset.reward[self._idx:] = reward[:first]
            self._dataset.reward[:rest] = reward[first:]
            self._dataset.next_state[self._idx:] = next_state[:first]
            self._dataset.next_state[:rest] = next_state[first:]
            self._dataset.absorbing[self._idx:] = absorbing[:first]
            self._dataset.absorbing[:rest] = absorbing[first:]
            self._dataset.last[self._idx:] = last[:first]
            self._dataset.last[:rest] = last[first:]
            if policy_state is not None:
                self._dataset.policy_state[self._idx:] = policy_state[:first]
                self._dataset.policy_state[:rest] = policy_state[first:]
                self._dataset.policy_next_state[self._idx:] = policy_next_state[:first]
                self._dataset.policy_next_state[:rest] = policy_next_state[first:]
            self._idx = rest

        return positions

    def _get_with_history(self, idxs):
        state_out = self._build_history(idxs, self._dataset.state)
        nstate_out = self._build_history(idxs, self._dataset.next_state)
        return state_out, nstate_out

    def _build_history(self, anchor_idxs, buffer):
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