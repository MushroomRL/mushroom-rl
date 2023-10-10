import numpy as np

from collections import defaultdict

from mushroom_rl.core.serialization import Serializable

from mushroom_rl.core._dataset_types import NumpyDataset


class Dataset(Serializable):
    def __init__(self, mdp_info, n_steps=None, n_episodes=None):
        assert (n_steps is not None and n_episodes is None) or (n_steps is None and n_episodes is not None)

        if n_steps is not None:
            n_samples = n_steps
        else:
            horizon = mdp_info.horizon
            assert np.isfinite(horizon)

            n_samples = horizon * n_episodes

        state_shape = (n_samples,) + mdp_info.observation_space.shape
        action_shape = (n_samples,) + mdp_info.action_space.shape
        reward_shape = (n_samples,)

        state_type = mdp_info.observation_space.data_type
        action_type = mdp_info.action_space.data_type

        self._info = defaultdict(list)
        self._data = NumpyDataset(state_type, state_shape, action_type, action_shape, reward_shape)
        self._gamma = mdp_info.gamma

        self._add_save_attr(
            _info='mushroom',
            _data='pickle',
            _gamma='primitive'
        )

    @classmethod
    def from_numpy(cls, states, actions, rewards, next_states, absorbings, lasts, gamma=0.99):
        """
        Creates a dataset of transitions from the provided arrays.

        Args:
            states (np.ndarray): array of states;
            actions (np.ndarray): array of actions;
            rewards (np.ndarray): array of rewards;
            next_states (np.ndarray): array of next_states;
            absorbings (np.ndarray): array of absorbing flags;
            lasts (np.ndarray): array of last flags.

        Returns:
            The list of transitions.

        """
        assert (len(states) == len(actions) == len(rewards)
                == len(next_states) == len(absorbings) == len(lasts))

        dataset = cls.__new__()
        dataset._gamma = gamma
        dataset._info = defaultdict(list)
        dataset._data = NumpyDataset.from_numpy(states, actions, rewards, next_states, absorbings, lasts, gamma)

        dataset._add_save_attr(
            _info='mushroom',
            _data='pickle',
            _gamma='primitive'
        )

        return dataset

    def append(self, step, info):
        self._data.append(*step[:6])
        self._append_info(info)

    def get_info(self, field, index=None):
        if index is None:
            return self._info[field]
        else:
            return self._info[field][index]

    def clear(self):
        self._info = defaultdict(list)
        self._data.clear()

    def get_view(self, index):
        dataset = self.copy()

        info_slice = defaultdict(list)
        for key in self._info.keys():
            info_slice[key] = self._info[key][index]

        dataset._info = info_slice
        dataset._data = self._data.get_view(index)

        return dataset

    def __getitem__(self, index):
        if isinstance(index, (slice, np.ndarray)):
            return self.get_view(index)
        elif isinstance(index, int) and index < len(self._data):
            return self._data[index]
        else:
            raise IndexError

    def __add__(self, other):
        result = self.copy()

        new_info = defaultdict(list)
        for key in self._info.keys():
            new_info[key] = self._info[key] + other.info[key]

        result._info = new_info
        result._data = self._data + other._data

        return result

    def __len__(self):
        return len(self._data)

    @property
    def state(self):
        return self._data.state

    @property
    def action(self):
        return self._data.action

    @property
    def reward(self):
        return self._data.reward

    @property
    def next_state(self):
        return self._data.next_state

    @property
    def absorbing(self):
        return self._data.absorbing

    @property
    def last(self):
        return self._data.last

    @property
    def info(self):
        return self._info

    @property
    def episodes_length(self):
        """
        Compute the length of each episode in the dataset.

        Args:
            dataset (list): the dataset to consider.

        Returns:
            A list of length of each episode in the dataset.

        """
        lengths = list()
        l = 0
        for sample in self:
            l += 1
            if sample[-1] == 1:
                lengths.append(l)
                l = 0

        return lengths

    @property
    def undiscounted_return(self):
        return self.compute_J()

    @property
    def discounted_return(self):
        return self.compute_J(self._gamma)

    def parse(self, index=None):
        """
        Return the dataset as set of arrays.

        Args (index, [int, slice]): index or slicee of dataset to be selected

        Returns:
            A tuple containing the arrays that define the dataset, i.e. state, action, next state, absorbing and last

        """
        if index is None:
            return self.state, self.action, self.reward, self.next_state, self.absorbing, self.last
        else:
            return self.state[index], self.action[index], self.reward[index], self.next_state[index], \
                   self.absorbing[index], self.last[index]

    def select_first_episodes(self, n_episodes):
        """
        Return the first ``n_episodes`` episodes in the provided dataset.

        Args:
            dataset (list): the dataset to consider;
            n_episodes (int): the number of episodes to pick from the dataset;

        Returns:
            A subset of the dataset containing the first ``n_episodes`` episodes.

        """
        assert n_episodes > 0, 'Number of episodes must be greater than zero.'

        last_idxs = np.argwhere(self.last==True).ravel()
        return self[:last_idxs[n_episodes - 1] + 1]

    def select_random_samples(self, n_samples):
        """
        Return the randomly picked desired number of samples in the provided
        dataset.

        Args:
            dataset (list): the dataset to consider;
            n_samples (int): the number of samples to pick from the dataset.

        Returns:
            A subset of the dataset containing randomly picked ``n_samples``
            samples.

        """
        assert n_samples >= 0, 'Number of samples must be greater than or equal to zero.'

        if n_samples == 0:
            return np.array([[]])

        idxs = np.random.randint(len(self), size=n_samples)

        return self[idxs]

    def get_init_states(self):
        """
        Get the initial states of a dataset

        Args:
            dataset (list): the dataset to consider.

        Returns:
            An array of initial states of the considered dataset.

        """
        pick = True
        x_0 = list()
        for step in self:
            if pick:
                x_0.append(step[0])
            pick = step[-1]
        return np.array(x_0)

    def compute_J(self, gamma=1.):
        """
        Compute the cumulative discounted reward of each episode in the dataset.

        Args:
            dataset (list): the dataset to consider;
            gamma (float, 1.): discount factor.

        Returns:
            The cumulative discounted reward of each episode in the dataset.

        """
        js = list()

        j = 0.
        episode_steps = 0
        for i in range(len(self)):
            j += gamma ** episode_steps * self.reward[i]
            episode_steps += 1
            if self.last[i] or i == len(self) - 1:
                js.append(j)
                j = 0.
                episode_steps = 0

        if len(js) == 0:
            return [0.]
        return js

    def compute_metrics(self, gamma=1.):
        """
        Compute the metrics of each complete episode in the dataset.

        Args:
            dataset (list): the dataset to consider;
            gamma (float, 1.): the discount factor.

        Returns:
            The minimum score reached in an episode,
            the maximum score reached in an episode,
            the mean score reached,
            the median score reached,
            the number of completed episodes.

            If no episode has been completed, it returns 0 for all values.

        """
        for i in reversed(range(len(self))):
            if self.last[i]:
                i += 1
                break

        dataset = self[:i]

        if len(dataset) > 0:
            J = dataset.compute_J(gamma)
            return np.min(J), np.max(J), np.mean(J), np.median(J), len(J)
        else:
            return 0, 0, 0, 0, 0

    def _append_info(self, step_info):
        for key, value in step_info.items():
            self._info[key].append(value)



