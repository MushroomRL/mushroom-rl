import numpy as np

from mushroom_rl.core import ArrayBackend
from mushroom_rl.rl_utils.parameters import to_parameter
from mushroom_rl.utils.sum_tree import SumTree

from mushroom_rl.rl_utils.replay_memory.replay_memory import ReplayMemory


class PrioritizedReplayMemory(ReplayMemory):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, alpha, beta, epsilon=.01,
                 history_manager=None, n_steps_return=1, store_policy_state=False, return_extra=False):
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
            return_extra (bool, False): whether :meth:`get` appends the ``extra_data`` dictionary of the history
                windows not delivered in-band in the state, inserted just before the ``idxs``/``is_weight`` pair.

        """
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon

        super().__init__(mdp_info, agent_info, initial_size, max_size, history_manager, n_steps_return,
                         store_policy_state, return_extra)

        self._add_save_attr(
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='mushroom'
        )

    def add(self, dataset, p):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): dataset whose transitions will be added to the replay memory;
            p (Array): priority of each sample in the dataset.

        """
        assert not self._dataset.is_stateful or dataset.is_stateful, \
            "The replay memory is configured to store the policy state, but the dataset does not provide it."

        if self._n_steps_return > 1:
            state, action, reward, next_state, absorbing, last = dataset.parse(to=self._agent_info.backend)
            policy_state, policy_next_state = (dataset.parse_policy_state(to=self._agent_info.backend)
                                               if self._dataset.is_stateful else (None, None))
            result = self._compute_n_step_return(state, action, reward, next_state, absorbing, last,
                                                 policy_state, policy_next_state, priority=p)
            if result is None:
                return
            dataset, p = result
        else:
            dataset = dataset.to_backend(self._agent_info.backend)

        positions = self._write_to_buffer(dataset)
        tree_idxs = ArrayBackend.convert(positions, to='numpy') + self._max_size - 1
        self._tree.update(tree_idxs, ArrayBackend.convert(p, to='numpy'))
        self._mask_write_head()

    def _mask_write_head(self):
        """
        Mask the leaves of the ``max_reach`` oldest samples of a full buffer so that they are never sampled: an anchor
        at one of these positions would rebuild a window across the write head, stitching together entries from
        unrelated trajectory segments. Their true priorities need not be preserved, as each masked leaf is overwritten
        with a fresh priority before the write head laps back to it. A no-op until the buffer is full or when no history
        is used.

        """
        if self._max_reach > 0 and self._full:
            positions = (self._idx + np.arange(self._max_reach)) % self._max_size
            self._tree.mask(positions + self._max_size - 1)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples, followed by the tree indices and importance-sampling weights of the drawn
            transitions. When ``return_extra`` is set, the ``extra_data`` dictionary assembled by the history manager is
            inserted just before this trailing pair.

        """
        idxs = np.zeros(n_samples, dtype=int)
        priorities = np.zeros(n_samples, dtype=float)

        total_p = self._tree.total_p
        segment = total_p / n_samples
        samples = np.random.uniform(
            np.arange(n_samples) * segment,
            np.arange(1, n_samples + 1) * segment
        )

        for i, s in enumerate(samples):
            idxs[i], priorities[i] = self._tree.get(s)

        is_weight = (self.size * priorities / total_p) ** -self._beta()
        is_weight /= is_weight.max()

        data_idxs = ArrayBackend.convert(idxs - self._max_size + 1, to=self._agent_info.backend)
        batch = self._dataset[data_idxs]
        state, action, reward, next_state, absorbing, last = batch.parse()

        extra = dict()
        if self._history_manager is not None:
            state, next_state, extra = self._history_manager.build_transition_windows_circular_buffer(
                self._dataset.state, self._dataset.next_state, self._dataset.action,
                self._dataset.last, data_idxs, len(self._dataset), self._full, self._max_size)

        out = [state, action, reward, next_state, absorbing, last]
        if self._dataset.is_stateful:
            out += list(batch.parse_policy_state())
        if self._return_extra:
            out.append(extra)
        out += [idxs, is_weight]

        return tuple(out)

    def reset(self):
        super().reset()
        self._tree = SumTree(self._max_size)

    def update(self, error, idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            error (Array): errors to consider to compute the priorities;
            idx (Array): indexes of the transitions in the dataset.

        """
        error = ArrayBackend.convert(error, to='numpy')
        idx = ArrayBackend.convert(idx, to='numpy')
        p = self._get_priority(error)
        self._tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._tree.max_p if self.initialized else 1.
