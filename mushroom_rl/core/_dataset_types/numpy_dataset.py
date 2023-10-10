import numpy as np

from mushroom_rl.core.serialization import Serializable


class NumpyDataset(Serializable):
    def __init__(self, state_type, state_shape, action_type, action_shape, reward_shape):
        flags_shape = (action_shape[0],)

        self._state_type = state_type
        self._action_type = action_type

        self._states = np.empty(state_shape, dtype=self._state_type)
        self._actions = np.empty(action_shape, dtype=self._action_type)
        self._rewards = np.empty(reward_shape, dtype=float)
        self._next_states = np.empty(state_shape, dtype=self._state_type)
        self._absorbing = np.empty(flags_shape, dtype=bool)
        self._last = np.empty(flags_shape, dtype=bool)
        self._len = 0

        self._add_save_attr(
            _states='numpy',
            _actions='numpy',
            _rewards='numpy',
            _next_states='numpy',
            _absorbing='numpy',
            _last='numpy',
            _len='primitive'
        )

    def __len__(self):
        return self._len

    def append(self, state, action, reward, next_state, absorbing, last):
        i = self._len

        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._absorbing[i] = absorbing
        self._last[i] = last

        self._len += 1

    def clear(self):
        self._states = np.empty_like(self._states)
        self._actions = np.empty_like(self._actions)
        self._rewards = np.empty_like(self._rewards)
        self._next_states = np.empty_like(self._next_states)
        self._absorbing = np.empty_like(self._absorbing)
        self._last = np.empty_like(self._last)
        self._len = 0

    def get_view(self, index):
        view = self.copy()

        view._states = self._states[index, ...]
        view._actions = self._actions[index, ...]
        view._rewards = self._rewards[index, ...]
        view._next_states = self._next_states[index, ...]
        view._absorbing = self._absorbing[index, ...]
        view._last = self._last[index, ...]
        view._len = view._states.shape[0]

        return view

    def __getitem__(self, index):
        return self._states[index], self._actions[index], self._rewards[index], self._next_states[index], \
               self._absorbing[index], self._last[index]

    def __add__(self, other):
        result = self.copy()

        result._states = np.concatenate((self.state, other.state))
        result._actions = np.concatenate((self.action, other.action))
        result._rewards = np.concatenate((self.reward, other.reward))
        result._next_states = np.concatenate((self.next_state, other.next_state))
        result._absorbing = np.concatenate((self.absorbing, other.absorbing))
        result._last = np.concatenate((self.last, other.last))
        result._len = len(self) + len(other)

        return result

    @property
    def state(self):
        return self._states

    @property
    def action(self):
        return self._actions

    @property
    def reward(self):
        return self._rewards

    @property
    def next_state(self):
        return self._next_states

    @property
    def absorbing(self):
        return self._absorbing

    @property
    def last(self):
        return self._last
