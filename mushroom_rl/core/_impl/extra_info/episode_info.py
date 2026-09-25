from mushroom_rl.core.mushroom_object import MushroomObject
from .step_info import StepInfo


class EpisodeInfo(MushroomObject):
    """
    Container for one entry per episode and per environment, such as the information an environment reports
    when it resets or the policy parameters an episodic agent draws. The entries of every environment are kept
    apart, and can be turned into a flat dictionary of arrays when they are dictionaries.

    """
    def __init__(self, n_envs, backend, device=None, vectorized=None):
        """
        Constructor.

        Args:
            n_envs (int): Number of parallel environments;
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on;
            vectorized (bool, None): whether the appended entries cover every environment rather than a single
                one. If None, it defaults to ``n_envs > 1``.

        """
        self._vectorized = n_envs > 1 if vectorized is None else vectorized
        self._backend = backend
        self._device = device
        self._episodes = [[] for _ in range(n_envs)]
        self._parsed = None
        self._target_backend = None
        self._target_device = None

        self._add_save_attr(
            _vectorized='primitive',
            _backend='primitive',
            _device='primitive',
            _episodes='pickle',
            _parsed='none',
            _target_backend='primitive',
            _target_device='primitive'
        )

    def __add__(self, other):
        """
        Combine two EpisodeInfo objects into a new one.

        Args:
            other (EpisodeInfo): EpisodeInfo which will be combined with this one.

        Returns:
            A new EpisodeInfo holding the episodes of both, environment by environment.

        """
        info = self.copy()
        info += other

        return info

    def __iadd__(self, other):
        """
        In-place append: extend every environment's episodes with the ones of another EpisodeInfo.

        Args:
            other (EpisodeInfo): EpisodeInfo whose episodes will be appended.

        """
        assert self.n_envs == other.n_envs

        for env, episodes in enumerate(other._episodes):
            self._episodes[env] += episodes
        self._parsed = None

        return self

    def __len__(self):
        return sum(len(episodes) for episodes in self._episodes)

    def append(self, entry, mask=None):
        """
        Append the entries of the environments the mask selects, or a single entry when no mask is given.

        Args:
            entry (dict, list, Array): the entry to append, covering every environment when a mask is given
                and belonging to the only environment otherwise;
            mask (Array, None): boolean mask selecting the environments to append for.

        """
        if mask is None:
            self._append_single(entry)
        else:
            self._append_masked(entry, mask)

        self._parsed = None

    def parse(self):
        """
        Turn the appended episodes into a flat dictionary of arrays, one entry per key.

        Returns:
            A flat dictionary containing an array for every key of the episode information, with the
            environments one after the other.

        """
        if self._parsed is None:
            info = StepInfo(1, self._backend, self._device, vectorized=False)
            for entry in self._flat_entries():
                info.append(entry)

            if self._target_backend is not None:
                info = info.to_backend(self._target_backend, self._target_device)

            self._parsed = info.parse()

        return self._parsed

    def to_backend(self, backend, device=None):
        """
        Copy the episode information, to be parsed in another backend.

        Args:
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on, or ``None`` for the default one.

        Returns:
            A new EpisodeInfo holding the same episodes, parsed in the given backend.

        """
        info = self.copy()

        if backend != self._backend or device != self._device:
            info._target_backend = backend
            info._target_device = device
        else:
            info._target_backend = None
            info._target_device = None

        return info

    def flatten(self):
        """
        Returns:
            A single-environment EpisodeInfo holding the episodes of every environment, one after the other.

        """
        info = EpisodeInfo(1, self._backend, self._device, vectorized=False)
        info._target_backend = self._target_backend
        info._target_device = self._target_device
        info._episodes[0].extend(self._flat_entries())

        return info

    def empty(self):
        """
        Returns:
            A new EpisodeInfo holding no episodes, shaped like this one.

        """
        return EpisodeInfo(self.n_envs, self._backend, self._device, vectorized=self._vectorized)

    def copy(self):
        """
        Returns:
            A new EpisodeInfo holding the same episodes.

        """
        info = self.empty()
        info._episodes = [episodes.copy() for episodes in self._episodes]
        info._target_backend = self._target_backend
        info._target_device = self._target_device

        return info

    def clear(self):
        """
        Drop the episodes of every environment.

        """
        self._episodes = [[] for _ in range(self.n_envs)]
        self._parsed = None

    @property
    def episodes(self):
        """
        Returns:
            The episodes of every environment, one list per environment, or the episodes of the only
            environment when the entries are not vectorized.

        """
        return self._episodes if self._vectorized else self._episodes[0]

    @property
    def n_envs(self):
        return len(self._episodes)

    def _append_single(self, entry):
        """
        Append the entry of the only environment.

        Args:
            entry: the entry to append.

        """
        self._episodes[0].append(entry)

    def _append_masked(self, entry, mask):
        """
        Append the entries of the environments the mask selects.

        Args:
            entry (dict, list, Array): the entry covering every environment;
            mask (Array): boolean mask selecting the environments to append for.

        """
        for env in range(self.n_envs):
            if mask[env]:
                self._episodes[env].append(self._entry(entry, env))

    def _flat_entries(self):
        """
        Returns:
            A flat list holding the episodes of every environment, one environment after the other.

        """
        flat = list()
        for episodes in self._episodes:
            flat += episodes

        return flat

    def _entry(self, entry, env):
        """
        Extract the entry of one environment from a batched entry.

        Args:
            entry (dict, list, Array): the appended entry;
            env (int): the environment to take the entry of.

        Returns:
            The entry of the given environment.

        """
        if isinstance(entry, dict):
            return {key: value[env] for key, value in entry.items()}

        return entry[env]
