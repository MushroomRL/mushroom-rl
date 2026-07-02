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
                 history_length=1, n_steps_return=1, store_policy_state=False):
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
            history_length (int, 1): number of consecutive observations per sample;
            n_steps_return (int, 1): number of steps used for the n-step return.

        """
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon

        super().__init__(mdp_info, agent_info, initial_size, max_size, history_length, n_steps_return,
                         store_policy_state)

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

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        idxs = np.zeros(n_samples, dtype=int)
        priorities = np.zeros(n_samples, dtype=float)
        data_idxs = np.zeros(n_samples, dtype=int)

        total_p = self._tree.total_p
        segment = total_p / n_samples
        samples = np.random.uniform(
            np.arange(n_samples) * segment,
            np.arange(1, n_samples + 1) * segment
        )

        for i, s in enumerate(samples):
            idx, p = self._tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            data_idxs[i] = idx - self._max_size + 1

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        data_idxs = ArrayBackend.convert(data_idxs, to=self._agent_info.backend)

        if self._history_length > 1:
            state_out, nstate_out = self._get_with_history(data_idxs)
            batch = self._dataset[data_idxs]
            _, action, reward, _, absorbing, last = batch.parse()
            return state_out, action, reward, nstate_out, absorbing, last, idxs, is_weight
        elif self._dataset.is_stateful:
            return *self._dataset[data_idxs].parse(), \
                   *self._dataset[data_idxs].parse_policy_state(), idxs, is_weight
        else:
            return *self._dataset[data_idxs].parse(), idxs, is_weight

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
