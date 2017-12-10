import numpy as np


class Buffer(object):
    """
    Utility class to manage Atari games states. It is used to build the current
    state of the agent to provide to the policy.

    """
    def __init__(self, size):
        """
        Constructor.

        Args:
            size (int): number of elements for each state (e.g. number of frames
                for each Atari state).

        """
        self._size = size

        self._buf = [None] * self._size

    def add(self, sample):
        """
        Add an element in the buffer as a queue.

        Args:
            sample (np.ndarray): the element to add to the buffer queue.

        """
        self._buf.append(sample)
        self._buf = self._buf[1:]

    def get(self):
        """
        Returns:
            The elements in the buffer.

        """
        s = np.empty(self._buf[0].shape + (self._size,), dtype=np.float32)
        for i in xrange(self._size):
            s[..., i] = self._buf[i]

        return s

    @property
    def size(self):
        """
        Returns:
            The number of elements that the buffer contains.

        """
        return self._size


class ReplayMemory(object):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    In the case of the Atari games, the replay memory stores each single frame,
    then returns a state composed of a provided number of concatenated frames.
    In Atari games, this helps to provide information about the history of the
    game.

    """
    def __init__(self, mdp_info, initial_size, max_size, history_length=1):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): an object containing the info of the
                environment;
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            history_length (int, 1): number of frames to concatenate to compose
                the state.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._history_length = history_length
        self._idx = 0
        self._full = False

        self._observation_shape = tuple(
            [self._max_size]) + mdp_info.observation_space.shape
        self._action_shape = (self._max_size, mdp_info.action_space.shape[0])

        self._states = np.ones(self._observation_shape, dtype=np.float32)
        self._actions = np.ones(self._action_shape, dtype=np.float32)
        self._rewards = np.ones(self._max_size, dtype=np.float32)
        self._absorbing = np.ones(self._max_size, dtype=np.bool)
        self._last = np.ones(self._max_size, dtype=np.bool)

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory.

        """
        for i in xrange(len(dataset)):
            self._states[self._idx, ...] = dataset[i][0]
            self._actions[self._idx, ...] = dataset[i][1]
            self._rewards[self._idx, ...] = dataset[i][2]
            self._absorbing[self._idx, ...] = dataset[i][4]
            self._last[self._idx, ...] = dataset[i][5]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of states to return.

        Returns:
            The requested number of states.

        """
        idxs = np.random.randint(self.size, size=n_samples)

        return self.get_idxs(idxs)

    def generator(self, batch_size, indexes=None):
        """
        Generator to iterate over the replay memory states.

        Args:
             batch_size (int): number of elements of the batch to be generated;
             indexes (list): indexes to use to extract states;

        Yields:
            A batch composed of the required number of states.

        """
        indexes = np.arange(self.size) if indexes is None else indexes
        n_batches = int(np.ceil(indexes.size / float(batch_size)))
        np.random.shuffle(indexes)
        batches = [(i * batch_size, min(indexes.size, (
            i + 1) * batch_size)) for i in xrange(n_batches)]
        for (batch_start, batch_end) in batches:
            yield self.get_idxs(indexes[batch_start:batch_end])

    def get_idxs(self, idxs):
        """
        Returns the states a the provided indexes.

        Args:
            idxs (list): the indexes of the states to return.

        Returns:
            The states at the provided indexes.

        """
        if not self._full and np.any(idxs < self._history_length):
            idxs[np.argwhere(
                idxs < self._history_length).ravel()] += self._history_length

        s = self._get_state(idxs - 1)
        ss = self._get_state(idxs)

        return s, self._actions[idxs - 1, ...], self._rewards[idxs - 1, ...],\
            ss, self._absorbing[idxs - 1, ...], self._last[idxs - 1, ...]

    def _get_state(self, idxs):
        """
        Build a state from the elements in the replay memory. A state is
        composed of the elements at the provided index and the following
        `history_length` elements.

        Args:
            idxs (list): the indexes of the states to return.

        Returns:
            The states built from the elements at the provided indexes.

        """
        s = np.empty((idxs.size,) + self._states.shape[1:] + (
            self._history_length,), dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx >= self._history_length - 1:
                for k in xrange(self._history_length):
                    s[j, ..., self._history_length - 1 - k] = self._states[
                        idx - k, ...]
            else:
                indexes = [(idx - i) % self.size for i in
                           reversed(range(self._history_length))]
                for k, index in enumerate(indexes):
                    s[j, ..., k] = self._states[index, ...]

        return s

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.ones(self._observation_shape, dtype=np.float32)
        self._actions = np.ones(self._action_shape, dtype=np.float32)
        self._rewards = np.ones(self._max_size, dtype=np.float32)
        self._absorbing = np.ones(self._max_size, dtype=np.bool)
        self._last = np.ones(self._max_size, dtype=np.bool)

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
