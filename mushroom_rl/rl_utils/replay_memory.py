import numpy as np
import torch

from mushroom_rl.core import Dataset, Serializable
from mushroom_rl.rl_utils.parameters import to_parameter


class ReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size):

        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False

        assert mdp_info.backend in ["numpy", "torch"], f"{mdp_info.backend} backend currently not supported in " \
                                                       f"the replay memory class."

        self.dataset = Dataset(mdp_info=mdp_info, agent_info=agent_info, n_steps=max_size, n_envs=1)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive!',
            _full='primitive!',
            dataset='mushroom',
        )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """

        assert n_steps_return > 0
        assert self.dataset.is_stateful == dataset.is_stateful

        # todo: implement vectorized n_step_return to avoid loop
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset.reward[i]
            j = 0
            while j < n_steps_return - 1:
                if dataset.last[i + j]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset.reward[i + j]
            else:
                if self._full:
                    self.dataset.state[self._idx] = self._backend.copy(dataset.state[i])
                    self.dataset.action[self._idx] = self._backend.copy(dataset.action[i])
                    self.dataset.reward[self._idx] = self._backend.copy(reward)

                    self.dataset.next_state[self._idx] = self._backend.copy(dataset.next_state[i + j])
                    self.dataset.absorbing[self._idx] = self._backend.copy(dataset.absorbing[i + j])
                    self.dataset.last[self._idx] = self._backend.copy(dataset.last[i + j])

                    if self.dataset.is_stateful:
                        self.dataset.policy_state[self._idx] = self._backend.copy(dataset.policy_state[i])
                        self.dataset.policy_next_state[self._idx] = self._backend.copy(dataset.policy_next_state[i + j])

                else:

                    sample = [self._backend.copy(dataset.state[i]),
                              self._backend.copy(dataset.action[i]),
                              self._backend.copy(reward),
                              self._backend.copy(dataset.next_state[i + j]),
                              self._backend.copy(dataset.absorbing[i + j]),
                              self._backend.copy(dataset.last[i + j])]

                    if self.dataset.is_stateful:
                        sample += [self._backend.copy(dataset.policy_state[i]),
                                   self._backend.copy(dataset.policy_next_state[i+j])]
                        self.dataset.append(sample, {})
                    else:
                        self.dataset.append(sample, {})

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

                i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        idxs = self.dataset.array_backend.randint(0, len(self.dataset), (n_samples,))

        default = (self.dataset.array_backend.copy(self.dataset.state[idxs]),
                   self.dataset.array_backend.copy(self.dataset.action[idxs]),
                   self.dataset.array_backend.copy(self.dataset.reward[idxs]),
                   self.dataset.array_backend.copy(self.dataset.next_state[idxs]),
                   self.dataset.array_backend.copy(self.dataset.absorbing[idxs]),
                   self.dataset.array_backend.copy(self.dataset.last[idxs]))

        if self.dataset.is_stateful:
            return *default, self.dataset.array_backend.copy(self.dataset.policy_state[idxs]),\
                   self.dataset.array_backend.copy(self.dataset.policy_next_state[idxs])
        else:
            return default

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self.dataset.clear()

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    @property
    def _backend(self):
        return self.dataset.array_backend

    def _post_load(self):
        if self._full is None:
            self.reset()


class SequenceReplayMemory(ReplayMemory):
    """
    This class extend the base replay memory to allow sampling sequences of a certain length. This is useful for
    training recurrent agents or agents operating on a window of states etc.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, truncation_length):
        self._truncation_length = truncation_length
        self._action_space_shape = mdp_info.action_space.shape

        super(SequenceReplayMemory, self).__init__(mdp_info, agent_info, initial_size, max_size)

        self._add_save_attr(
            _truncation_length='primitive',
            _action_space_shape='primitive'
        )

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """

        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()         # previous actions
        lengths = list()    # lengths of the sequences

        for i in np.random.randint(self.size, size=n_samples):

            with torch.no_grad():

                # determine the begin of a sequence
                begin_seq = np.maximum(i - self._truncation_length + 1, 0)
                end_seq = i + 1

                # the sequence can contain more than one trajectory, check to only include one
                lasts_absorbing = self.dataset.last[begin_seq: i]
                begin_traj = self._backend.where(lasts_absorbing > 0)
                more_than_one_traj = len(*begin_traj) > 0
                if more_than_one_traj:
                    # take the beginning of the last trajectory
                    begin_seq = begin_seq + begin_traj[0][-1] + 1

                # get data and apply padding if needed
                states = self._backend.copy(self.dataset.state[begin_seq:end_seq])
                next_states = self._backend.copy(self.dataset.next_state[begin_seq:end_seq])
                action_seq = self._backend.copy(self.dataset.action[begin_seq:end_seq])
                if more_than_one_traj or begin_seq == 0 or self.dataset.last[begin_seq-1]:
                    prev_actions = self._backend.copy(self.dataset.action[begin_seq:end_seq - 1])
                    init_prev_action = self._backend.zeros(1, *self._action_space_shape)
                    if len(prev_actions) == 0:
                        prev_actions = init_prev_action
                    else:
                        prev_actions = self._backend.concatenate([init_prev_action, prev_actions])
                else:
                    prev_actions = self._backend.copy(self.dataset.action[begin_seq - 1:end_seq - 1])

                length_seq = len(states)

                s.append(self._backend.expand_dims(self._add_padding(states), dim=0))
                a.append(self._backend.expand_dims(self._add_padding(action_seq), dim=0))
                r.append(self._backend.expand_dims(self._backend.copy(self.dataset.reward[i]), dim=0))
                ss.append(self._backend.expand_dims(self._add_padding(next_states), dim=0))
                ab.append(self._backend.expand_dims(self._backend.copy(self.dataset.absorbing[i]), dim=0))
                last.append(self._backend.expand_dims(self._backend.copy(self.dataset.last[i]), dim=0))
                pa.append(self._backend.expand_dims(self._add_padding(prev_actions), dim=0))
                lengths.append(length_seq)

        return self._backend.concatenate(s), self._backend.concatenate(a), self._backend.concatenate(r),\
               self._backend.concatenate(ss), self._backend.concatenate(ab), self._backend.concatenate(last),\
               self._backend.concatenate(pa), lengths

    def _add_padding(self, array):
        return self._backend.concatenate([array, self._backend.zeros(
            self._truncation_length - array.shape[0], array.shape[1])])


class SumTree(object):
    """
    This class implements a sum tree data structure.
    This is used, for instance, by ``PrioritizedReplayMemory``.

    """
    def __init__(self, max_size):
        """
        Constructor.

        Args:
            max_size (int): maximum size of the tree.

        """
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._data = [None for _ in range(max_size)]
        self._idx = 0
        self._full = False

    def add(self, dataset, priority, n_steps_return, gamma):
        """
        Add elements to the tree.

        Args:
            dataset (list): list of elements to add to the tree;
            priority (np.ndarray): priority of each sample in the dataset;
            n_steps_return (int): number of steps to consider for computing n-step return;
            gamma (float): discount factor for n-step return.

        """
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]

            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                d = list(dataset[i])
                d[2] = reward
                d[3] = dataset[i + j][3]
                d[4] = dataset[i + j][4]
                d[5] = dataset[i + j][5]
                idx = self._idx + self._max_size - 1

                self._data[self._idx] = d
                self.update([idx], [priority[i]])

                self._idx += 1
                if self._idx == self._max_size:
                    self._idx = 0
                    self._full = True

                i += 1

    def get(self, s):
        """
        Returns the provided number of states from the replay memory.

        Args:
            s (float): the value of the samples to return.

        Returns:
            The requested sample.

        """
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1

        return idx, self._tree[idx], self._data[data_idx]

    def update(self, idx, priorities):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            idx (np.ndarray): indexes of the transitions in the dataset;
            priorities (np.ndarray): priorities of the transitions.

        """
        for i, p in zip(idx, priorities):
            delta = p - self._tree[i]

            self._tree[i] = p
            self._propagate(delta, i)

    def _propagate(self, delta, idx):
        parent_idx = (idx - 1) // 2

        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(delta, parent_idx)

    def _retrieve(self, s, idx):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if self._tree[left] == self._tree[right]:
            return self._retrieve(s, np.random.choice([left, right]))

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def size(self):
        """
        Returns:
            The current size of the tree.

        """
        return self._idx if not self._full else self._max_size

    @property
    def max_p(self):
        """
        Returns:
            The maximum priority among the ones in the tree.

        """
        return self._tree[-self._max_size:].max()

    @property
    def total_p(self):
        """
        Returns:
            The sum of the priorities in the tree, i.e. the value of the root
            node.

        """
        return self._tree[0]


class PrioritizedReplayMemory(Serializable):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, initial_size, max_size, alpha, beta, epsilon=.01):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay
                memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon

        self._tree = SumTree(max_size)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='pickle!'
        )

    def add(self, dataset, p, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            p (np.ndarray): priority of each sample in the dataset.
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0

        self._tree.add(dataset, p, n_steps_return, gamma)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]

        idxs = np.zeros(n_samples, dtype=int)
        priorities = np.zeros(n_samples)

        total_p = self._tree.total_p
        segment = total_p / n_samples

        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples + 1) * segment
        samples = np.random.uniform(a, b)
        for i, s in enumerate(samples):
            idx, p, data = self._tree.get(s)

            idxs[i] = idx
            priorities[i] = p
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i] = data
            states[i] = np.array(states[i])
            next_states[i] = np.array(next_states[i])

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            idxs, is_weight

    def update(self, error, idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            error (np.ndarray): errors to consider to compute the priorities;
            idx (np.ndarray): indexes of the transitions in the dataset.

        """
        p = self._get_priority(error)
        self._tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self._tree.size > self._initial_size

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._tree.max_p if self.initialized else 1.

    def _post_load(self):
        if self._tree is None:
            self._tree = SumTree(self._max_size)
