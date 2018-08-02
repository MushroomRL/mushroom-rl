import numpy as np


class ReplayMemory(object):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    In the case of the Atari games, the replay memory stores each single frame,
    then returns a state composed of a provided number of concatenated frames.
    In Atari games, this helps to provide information about the history of the
    game.

    """
    def __init__(self, mdp_info, initial_size, max_size):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): an object containing the info of the
                environment;
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self._observation_shape = tuple(
            [self._max_size]) + mdp_info.observation_space.shape
        self._action_shape = (self._max_size, mdp_info.action_space.shape[0])

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

        return np.stack([np.array(self._states[i]) for i in range(start, stop)]),\
            np.array([self._actions[i] for i in range(start, stop)]),\
            np.array([self._rewards[i] for i in range(start, stop)]),\
            np.stack([np.array(self._next_states[i]) for i in range(start, stop)]),\
            np.array([self._absorbing[i] for i in range(start, stop)]),\
            np.array([self._last[i] for i in range(start, stop)])

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
