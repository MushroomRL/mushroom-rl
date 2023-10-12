import torch

from mushroom_rl.core.serialization import Serializable


class TorchDataset(Serializable):
    def __init__(self, state_type, state_shape, action_type, action_shape, reward_shape):
        flags_len = action_shape[0]

        self._state_type = state_type
        self._action_type = action_type

        self._states = torch.empty(*state_shape, dtype=self._state_type)
        self._actions = torch.empty(*action_shape, dtype=self._action_type)
        self._rewards = torch.empty(*reward_shape, dtype=torch.float)
        self._next_states = torch.empty(*state_shape, dtype=self._state_type)
        self._absorbing = torch.empty(flags_len, dtype=torch.bool)
        self._last = torch.empty(flags_len, dtype=torch.bool)
        self._len = 0

        self._add_save_attr(
            _state_type='primitive',
            _action_type='primitive',
            _states='torch',
            _actions='torch',
            _rewards='torch',
            _next_states='torch',
            _absorbing='torch',
            _last='torch',
            _len='primitive'
        )

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts):
        dataset = cls.__new__(cls)

        dataset._state_type = states.dtype
        dataset._action_type = actions.dtype

        dataset._states = torch.as_tensor(states)
        dataset._actions = torch.as_tensor(actions)
        dataset._rewards = torch.as_tensor(rewards)
        dataset._next_states = torch.as_tensor(next_states)
        dataset._absorbing = torch.as_tensor(absorbings, dtype=torch.bool)
        dataset._last = torch.as_tensor(lasts, dtype=torch.bool)
        dataset._len = len(lasts)

        dataset._add_save_attr(
            _state_type='primitive',
            _action_type='primitive',
            _states='torch',
            _actions='torch',
            _rewards='torch',
            _next_states='torch',
            _absorbing='torch',
            _last='torch',
            _len='primitive'
        )

        return dataset

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
        self._states = torch.empty_like(self._states)
        self._actions = torch.empty_like(self._actions)
        self._rewards = torch.empty_like(self._rewards)
        self._next_states = torch.empty_like(self._next_states)
        self._absorbing = torch.empty_like(self._absorbing)
        self._last = torch.empty_like(self._last)

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

        result._states = torch.concatenate((self.state, other.state))
        result._actions = torch.concatenate((self.action, other.action))
        result._rewards = torch.concatenate((self.reward, other.reward))
        result._next_states = torch.concatenate((self.next_state, other.next_state))
        result._absorbing = torch.concatenate((self.absorbing, other.absorbing))
        result._last = torch.concatenate((self.last, other.last))
        result._len = len(self) + len(other)

        return result

    @property
    def state(self):
        return self._states[:len(self)]

    @property
    def action(self):
        return self._actions[:len(self)]

    @property
    def reward(self):
        return self._rewards[:len(self)]

    @property
    def next_state(self):
        return self._next_states[:len(self)]

    @property
    def absorbing(self):
        return self._absorbing[:len(self)]

    @property
    def last(self):
        return self._last[:len(self)]
