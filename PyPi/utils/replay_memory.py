import numpy as np

from PyPi.utils.dataset import parse_dataset


class ReplayMemory(object):
    def __init__(self, max_size, history_length=1):
        self._max_size = max_size
        self._history_length = history_length
        self._idx = 0
        self._full = False

    def initialize(self, mdp_info):
        observation_space = mdp_info['observation_space']
        action_space = mdp_info['action_space']

        observation_shape = tuple([self._max_size]) + observation_space.shape
        action_shape = (self._max_size, action_space.shape)

        self._states = np.ones(observation_shape, dtype=np.float32)
        self._actions = np.ones(action_shape, dtype=np.float32)
        self._rewards = np.ones(self._max_size, dtype=np.float32)
        self._absorbing = np.ones(self._max_size, dtype=np.bool)
        self._last = np.ones(self._max_size, dtype=np.bool)

    def add(self, dataset):
        next_idx = self._idx + len(dataset)
        assert next_idx <= self._max_size

        self._states[self._idx:next_idx, ...],\
            self._actions[self._idx:next_idx, ...],\
            self._rewards[self._idx:next_idx, ...], _,\
            self._absorbing[self._idx:next_idx, ...],\
            self._last[self._idx:next_idx, ...] = parse_dataset(dataset)

        self._idx = next_idx
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def get(self, n_samples):
        idxs = np.random.randint(self.size, size=n_samples)

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
            if idxs >= self._history_length - 1:
                s[j, ...] = self._states[
                    (idx - (self._history_length - 1)):(idx + 1), ...]
            else:
                indexes = [(idx - i) % self.size for i in
                           reversed(range(self._history_length))]
                s[j, ...] = self._states[indexes, ...]

        return s

    @property
    def size(self):
        return self._idx if not self._full else self._max_size
