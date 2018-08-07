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
        if self._current_sample_idx + n_samples >= len(self._sample_idxs):
            self._sample_idxs = np.random.choice(self.size, self.size,
                                                 replace=False)
            self._current_sample_idx = 0

        start = self._current_sample_idx
        stop = start + n_samples

        self._current_sample_idx = stop

        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        for i in self._sample_idxs[start:stop]:
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

        self._sample_idxs = np.random.choice(self._initial_size,
                                             self._initial_size,
                                             replace=False)
        self._current_sample_idx = 0

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size >= self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size
