import numpy as np


class Buffer(object):
    """
    Utility class to manage Atari games states.

    """
    def __init__(self, size):
        self._size = size

        self._buf = [None] * self._size

    def add(self, sample):
        self._buf.append(sample)
        self._buf = self._buf[1:]

    def get(self):
        s = np.empty(self._buf[0].shape + (self._size,), dtype=np.float32)
        for i in xrange(self._size):
            s[..., i] = self._buf[i]

        return s

    @property
    def size(self):
        return self._size


class ReplayMemory(object):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, mdp_info, initial_size, max_size, history_length=1):
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
        idxs = np.random.randint(self.size, size=n_samples)

        return self.get_idxs(idxs)

    def generator(self, batch_size, indexes=None):
        indexes = np.arange(self.size) if indexes is None else indexes
        n_batches = int(np.ceil(indexes.size / float(batch_size)))
        np.random.shuffle(indexes)
        batches = [(i * batch_size, min(indexes.size, (
            i + 1) * batch_size)) for i in xrange(n_batches)]
        for (batch_start, batch_end) in batches:
            yield self.get_idxs(indexes[batch_start:batch_end])

    def get_idxs(self, idxs):
        if not self._full and np.any(idxs < self._history_length):
            idxs[np.argwhere(
                idxs < self._history_length).ravel()] += self._history_length

        s = self._get_state(idxs - 1)
        ss = self._get_state(idxs)

        return s, self._actions[idxs - 1, ...], self._rewards[idxs - 1, ...],\
            ss, self._absorbing[idxs - 1, ...], self._last[idxs - 1, ...]

    def _get_state(self, idxs):
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
        self._idx = 0
        self._full = False
        self._states = np.ones(self._observation_shape, dtype=np.float32)
        self._actions = np.ones(self._action_shape, dtype=np.float32)
        self._rewards = np.ones(self._max_size, dtype=np.float32)
        self._absorbing = np.ones(self._max_size, dtype=np.bool)
        self._last = np.ones(self._max_size, dtype=np.bool)

    @property
    def initialized(self):
        return self.size > self._initial_size

    @property
    def size(self):
        return self._idx if not self._full else self._max_size
