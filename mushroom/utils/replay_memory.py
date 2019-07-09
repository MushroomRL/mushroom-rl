import numpy as np


class ReplayMemory(object):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self.reset()

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory.

        """
        for i in range(len(dataset)):
            self._states[self._idx] = dataset[i][0]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]
            self._next_states[self._idx] = dataset[i][3]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

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
        for i in np.random.choice(self.size, size=n_samples, replace=False):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last)

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._rewards = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]
        self._last = [None for _ in range(self._max_size)]

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


class SumTree(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._data = [None for _ in range(max_size)]
        self._idx = 0
        self._full = False

    def add(self, dataset, priority):
        for d, p in zip(dataset, priority):
            idx = self._idx + self._max_size - 1

            self._data[self._idx] = d
            self.update(idx, p)

            self._idx += 1
            if self._idx == self._max_size:
                self._idx = 0
                self._full = True

    def get(self, s):
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1

        return idx, self._tree[idx], self._data[data_idx]

    def update(self, idx, priorities):
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

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def size(self):
        return self._idx if not self._full else self._max_size

    @property
    def total_p(self):
        return self._tree[0]


class PrioritizedReplayMemory(object):
    def __init__(self, initial_size, max_size, alpha, beta, epsilon=.01):
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._tree = SumTree(max_size)

    def add(self, dataset, error):
        p = self._get_priority(error)
        self._tree.add(dataset, p)

    def get(self, n_samples):
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]

        segment = self._tree.total_p / n_samples
        idxs = np.zeros(n_samples)
        priorities = np.zeros(n_samples)

        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples) * segment
        samples = np.random.uniform(a, b, size=n_samples)
        for i, s in enumerate(samples):
            idx, p, data = self._tree.get(s)

            idxs[i] = idx
            priorities[i] = p
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i] = data

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        return (states, actions, rewards, next_states, absorbing, last), idxs,\
            is_weight

    def update(self, error, idx):
        p = self._get_priority(error)
        self._tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha
