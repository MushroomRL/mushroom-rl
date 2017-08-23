import numpy as np

from PyPi.utils.dataset import parse_dataset


class ReplayMemory(object):
    def __init__(self, initial_size, max_size):
        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False

    def initialize(self, mdp_info):
        observation_space = mdp_info['observation_space']
        action_space = mdp_info['action_space']

        observation_shape = tuple([self._max_size]) + observation_space.shape
        action_shape = (self._max_size, action_space.shape)

        self._states = np.empty(observation_shape, dtype=np.float32)
        self._actions = np.empty(action_shape, dtype=np.float32)
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_states = np.empty(observation_shape, dtype=np.float32)
        self._absorbing = np.empty(self._max_size, dtype=np.bool)
        self._last = np.empty(self._max_size, dtype=np.bool)

    def add(self, dataset):
        assert self._idx + len(dataset) <= self._max_size

        states, actions, rewards, next_states, absorbing, last = parse_dataset(
            dataset
        )

        self._states[self._idx:, ...] = states
        self._actions[self._idx:, ...] = actions
        self._rewards[self._idx:, ...] = rewards
        self._next_states[self._idx:, ...] = next_states
        self._absorbing[self._idx:, ...] = absorbing
        self._last[self._idx, ...] = last

        self._idx += len(dataset)
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def get(self, n_samples):
        idxs = np.random.randint(self.size, size=n_samples)

        return self._states[idxs, ...], self._actions[idxs, ...],\
            self._rewards[idxs, ...], self._next_states[idxs, ...],\
            self._absorbing[idxs, ...], self._last[idxs, ...]

    def is_ready(self):
        return self._idx >= self._initial_size or self._full

    @property
    def size(self):
        return self._idx if not self._full else self._max_size
