from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.utils.priority_sampler import PrioritySampler

from mushroom_rl.rl_utils.replay_memory.replay_memory import ReplayMemory


class PrioritizedReplayMemory(ReplayMemory):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, alpha, beta, epsilon=.01, history_manager=None,
                 n_steps_return=1, store_policy_state=False, return_extra=False):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities;
            history_manager (HistoryManager, None): the manager used by the agent to assemble the stacked observation,
                reused offline so that the stacked observation matches the one built online;
            n_steps_return (int, 1): number of steps used for the n-step return;
            store_policy_state (bool, False): whether the policy internal state is stored in the replay memory;
            return_extra (bool, False): whether :meth:`get` also returns the ``extra_data`` dictionary of the history
                windows not delivered in-band in the state.

        """
        self._alpha = alpha
        self._beta = Parameter.make(beta)
        self._epsilon = epsilon

        super().__init__(mdp_info, agent_info, initial_size, max_size, history_manager, n_steps_return,
                         store_policy_state, return_extra)

        self._add_save_attr(
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _sampler='mushroom'
        )

    def add(self, dataset, p=None):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): dataset whose transitions will be added to the replay memory;
            p (Array, None): priority of each sample in the dataset. When ``None``, each new transition is inserted
                with the current maximum priority (:attr:`max_priority`), so that it is sampled at least once before
                its priority is corrected from its temporal-difference error.

        """
        assert not self._dataset.is_stateful or dataset.is_stateful, \
            "The replay memory is configured to store the policy state, but the dataset does not provide it."

        if len(dataset) > 0:
            if p is None:
                p = self.max_priority * self._dataset.backend.ones(len(dataset), device=self._agent_info.device)

            start = self._dataset.write_head
            positions, relinked, orphans = self._dataset.append_replay_batch(dataset)
            self._sampler.update(positions, p)
            self._sync_sampler_mask(self._affected_window(start, len(positions), relinked, orphans))

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples, followed by the buffer positions and importance-sampling weights of the
            drawn transitions. When ``return_extra`` is set, the ``extra_data`` dictionary assembled by the history
            manager is inserted just before this trailing pair.

        """
        total_priority = self._sampler.total_priority
        idxs, priorities = self._sampler.sample(n_samples)

        importance_sampling_weights = (self.size * priorities / total_priority) ** -self._beta()
        importance_sampling_weights /= importance_sampling_weights.max()

        out = self._assemble_batch(idxs)
        out += [idxs, importance_sampling_weights]

        return tuple(out)

    def reset(self):
        super().reset()
        self._sampler = PrioritySampler(self._max_size, backend=self._agent_info.backend_name,
                                        device=self._agent_info.device)

    def update(self, errors, idxs):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            errors (Array): errors to consider to compute the priorities;
            idxs (Array): buffer positions of the transitions, as returned by :meth:`get`.

        """
        self._sampler.update(idxs, self._get_priority(errors))

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._sampler.max_priority if self.initialized else 1.

    def _sync_sampler_mask(self, window):
        """
        Mirror the sampling mask of ``window`` (see :meth:`_compute_mask`) into the priority sampler.

        Args:
            window: the buffer positions whose mask changed (see :meth:`_affected_window`), or ``None`` when no masking
                is in use.

        """
        if window is not None:
            self._sampler.set_mask(window, self._compute_mask(window))

    def _get_priority(self, error):
        return (abs(error) + self._epsilon) ** self._alpha
