from mushroom_rl.core import DatasetInfo, MushroomObject
from mushroom_rl.core.dataset import CircularDataset
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
        assert initial_size >= n_steps_return, \
            f"The initial size {initial_size} is smaller than the {n_steps_return} steps of the n-step return."

        self._initial_size = initial_size
        self._max_size = max_size
        self._history_manager = history_manager if history_manager is not None \
            else HistoryManager.default_streams(mdp_info, agent_info)
        self._n_steps_return = n_steps_return
        self._store_policy_state = store_policy_state
        self._return_extra = return_extra
        self._mdp_info = mdp_info
        self._agent_info = agent_info

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

        self._dataset.write(dataset)

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
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info,
                                                             self._store_policy_state)
        self._dataset = CircularDataset(dataset_info, self._max_size)

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._dataset.size

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
        if self._n_steps_return > 1:
            state, action, reward, next_state, absorbing, last, extra = self._history_manager.parse_nstep_history(
                ds, self._mdp_info.gamma, self._n_steps_return, anchor_idxs=idxs)
            anchor = extra.pop('anchor')
            endpoint = extra.pop('endpoint')
        else:
            state, action, reward, next_state, absorbing, last, extra = self._history_manager.parse_history(
                ds, anchor_idxs=idxs)
            anchor = endpoint = idxs

        policy_state = [ds.policy_state[anchor], ds.policy_next_state[endpoint]] if ds.is_stateful else []

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
            return backend.randint(0, size, (n_samples,), device=self._agent_info.device)
        idxs = backend.arange(0, size, device=self._agent_info.device)
        valid = idxs[~self._compute_mask(idxs)]
        return valid[backend.randint(0, len(valid), (n_samples,), device=self._agent_info.device)]

    def _affected_window(self, positions, relinked, orphans):
        """
        The buffer positions whose sampling mask can change after a batch was written at ``positions``: the newly
        written anchors, their forward n-step window (the ``n-1`` anchors ending in the new batch), the backward
        history reserve that trails the moved write head, for every open episode end the batch continued away
        from the write head the ``n-1`` anchors ending there and, for every stored step whose previous step was
        overwritten, that step and the steps whose history window reaches it. Every other entry keeps its mask.

        Args:
            positions: the buffer positions where the last batch was written;
            relinked (list): the buffer positions of the open episode ends the batch continued;
            orphans: the buffer positions of the stored steps whose previous step was overwritten.

        Returns:
            The affected buffer positions, or ``None`` when no masking is in use.

        """
        if self._history_manager.max_reach == 0 and self._n_steps_return == 1:
            return None

        backend = self._dataset.array_backend
        size = len(self._dataset)
        full = self._dataset.full
        history_reserve = self._history_manager.max_reach if full else 0
        window_length = len(positions) + (self._n_steps_return - 1) + history_reserve
        range_vec = backend.arange(0, window_length, device=self._agent_info.device)
        raw = (positions[0] - (self._n_steps_return - 1)) + range_vec
        window = raw % self._max_size if full else raw[(raw >= 0) & (raw < size)]
        if len(relinked) > 0 and self._dataset.links is not None:
            ends = backend.as_array(relinked, device=self._agent_info.device)
            reached = self._dataset.walk_back(ends, self._n_steps_return - 1)[0]
            window = backend.concatenate([window, reached.T.reshape(-1)])
        if len(orphans) > 0 and self._history_manager.max_reach > 0:
            reached = self._dataset.walk_forward(orphans, self._history_manager.max_reach - 1)[0]
            window = backend.concatenate([window, reached.T.reshape(-1)])
        return window

    def _compute_mask(self, anchor_idxs):
        """
        Compute the sampling mask for a batch of anchors: True where the anchor cannot be sampled because its n-step
        window would cross a truncation or the write head, or because its backward history window would cross the write
        head of a full buffer or reach an overwritten step.

        Args:
            anchor_idxs: buffer positions of the anchors to evaluate.

        Returns:
            The boolean mask (True = excluded from sampling) for the given anchors.

        """
        backend = self._dataset.array_backend
        mask = backend.zeros(len(anchor_idxs), dtype=bool, device=self._agent_info.device)
        ds = self._dataset
        if self._n_steps_return > 1:
            valid = self._history_manager.nstep_valid(ds.absorbing, ds.last, anchor_idxs, self._n_steps_return,
                                                      dataset=ds)
            mask = mask | ~valid
        if self._history_manager.max_reach > 0:
            if ds.links is not None:
                mask = mask | ds.history_cut(anchor_idxs, self._history_manager.max_reach)
            elif ds.full:
                mask = mask | ((anchor_idxs - ds.write_head) % self._max_size < self._history_manager.max_reach)
        return mask

    def _post_load(self):
        if self._dataset is None:
            self.reset()
