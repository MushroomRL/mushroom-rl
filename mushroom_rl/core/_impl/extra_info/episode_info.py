from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_backend_object import MushroomBackendObject

from .step_info import StepInfo


class EpisodeInfo(MushroomBackendObject):
    """
    Base class for storing data collected once per episode in each environment, such as the information returned
    on reset or the policy parameters drawn by an episodic agent. The data of each environment is kept separate,
    and each subclass defines how it is parsed into arrays.

    """
    def __init__(self, n_envs, backend, device=None, vectorized=None):
        """
        Constructor.

        Args:
            n_envs (int): number of parallel environments;
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on;
            vectorized (bool, None): whether the appended entries cover every environment rather than a single
                one. If None, it defaults to ``n_envs > 1``.

        """
        super().__init__(backend, device)

        self._n_envs = n_envs
        self._vectorized = n_envs > 1 if vectorized is None else vectorized
        self._blocks = self._initial_blocks()
        self._parsed = None
        self._target_backend_name = None
        self._target_device = None

        self._add_save_attr(
            _n_envs='primitive',
            _vectorized='primitive',
            _blocks='torch' if self._backend.get_backend_serialization() == 'torch' else 'pickle',
            _parsed='none',
            _target_backend_name='primitive',
            _target_device='none'
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

        if self._vectorized:
            for block in other._blocks:
                self._blocks[0] += block
        else:
            self._blocks += [block.copy() for block in other._blocks]
        self._parsed = None

        return self

    def __len__(self):
        return sum(1 if indices is None else len(indices) for block in self._blocks for _, indices in block)

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
        Parse the stored entries into arrays.

        Returns:
            The parsed entries, with the environments one after the other.

        """
        raise NotImplementedError

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
        device = ArrayBackend.get_array_backend(backend).check_device(device)

        if backend != self.backend_name or device != self._device:
            info._target_backend_name = backend
            info._target_device = device
        else:
            info._target_backend_name = None
            info._target_device = None

        return info

    def flatten(self):
        """
        Returns:
            A single-environment EpisodeInfo holding the episodes of every environment, one after the other.

        """
        info = self._create_instance(1, vectorized=False)
        info._target_backend_name = self._target_backend_name
        info._target_device = self._target_device
        info._blocks = [block.copy() for block in self._blocks if block]

        return info

    def empty(self):
        """
        Returns:
            A new EpisodeInfo holding no episodes, shaped like this one.

        """
        return self._create_instance(self.n_envs, vectorized=self._vectorized)

    def copy(self):
        """
        Returns:
            A new EpisodeInfo holding the same episodes.

        """
        info = self.empty()
        info._blocks = [block.copy() for block in self._blocks]
        info._target_backend_name = self._target_backend_name
        info._target_device = self._target_device

        return info

    def clear(self):
        """
        Drop the episodes of every environment.

        """
        self._blocks = self._initial_blocks()
        self._parsed = None

    @property
    def n_envs(self):
        return self._n_envs

    def _post_load(self, device):
        if self._target_backend_name is not None:
            target = ArrayBackend.get_array_backend(self._target_backend_name)
            self._target_device = target.check_device(device if target.supports_devices() else None)

    def _create_instance(self, n_envs, vectorized):
        """
        Args:
            n_envs (int): number of parallel environments;
            vectorized (bool): whether the appended entries cover every environment rather than a single one.

        Returns:
            A new, empty object of the same class, in the same backend and on the same device as this one.

        """
        raise NotImplementedError

    def _initial_blocks(self):
        """
        Returns:
            The blocks of an empty EpisodeInfo: a single one when the entries are vectorized, none otherwise.

        """
        return [[]] if self._vectorized else []

    def _append_single(self, entry):
        """
        Append the entry of the only environment.

        Args:
            entry: the entry to append.

        """
        self._append_chunk(entry, None)

    def _append_masked(self, entry, mask):
        """
        Append the entries of the environments the mask selects.

        Args:
            entry (dict, list, Array): the entry covering every environment;
            mask (Array): boolean mask selecting the environments to append for.

        """
        indices = ArrayBackend.get_array_backend_from(mask).nonzero(mask).reshape(-1)

        if len(indices) > 0:
            if isinstance(entry, dict):
                entry = {key: self._select(value, indices) for key, value in entry.items()}
            else:
                entry = self._select(entry, indices)

            self._append_chunk(entry, indices)

    def _append_chunk(self, entry, indices):
        """
        Store an appended entry: in the only block when the entries are vectorized, in a new block otherwise.

        Args:
            entry: the entry, restricted to the selected environments when ``indices`` is given;
            indices (Array, None): the environments the rows of the entry belong to, or ``None`` for a single
                entry of the first environment.

        """
        if self._vectorized:
            self._blocks[0].append((entry, indices))
        else:
            self._blocks.append([(entry, indices)])

    def _flat_entries(self):
        """
        Returns:
            A flat list holding the episodes of every environment, one environment after the other within each
            block, and the blocks one after the other.

        """
        flat = list()
        for block in self._blocks:
            episodes = self._grouped_entries(block)
            for env in sorted(episodes):
                flat += episodes[env]

        return flat

    def _grouped_entries(self, block):
        """
        Args:
            block (list): the chunks of a block.

        Returns:
            A dictionary mapping every environment of the block to its episodes, in the order they were appended.

        """
        episodes = dict()
        for entry, indices in block:
            if indices is None:
                episodes.setdefault(0, []).append(entry)
            else:
                for row, env in enumerate(indices.tolist()):
                    episodes.setdefault(env, []).append(self._row(entry, row))

        return episodes

    @staticmethod
    def _select(value, indices):
        """
        Args:
            value (list, Array): a value with one element per environment;
            indices (Array): the environments to select.

        Returns:
            The elements of the selected environments, in the same type as ``value``.

        """
        backend = ArrayBackend.get_array_backend_from(value)

        if backend.get_backend_name() == 'list':
            return [value[env] for env in indices.tolist()]

        device = backend.get_device(value)
        return value[backend.convert(indices, device=device)]

    @staticmethod
    def _row(entry, row):
        """
        Args:
            entry (dict, list, Array): an entry restricted to the selected environments;
            row (int): the position of an environment among the selected ones.

        Returns:
            The entry of that environment.

        """
        if isinstance(entry, dict):
            return {key: value[row] for key, value in entry.items()}

        return entry[row]


class ResetEpisodeInfo(EpisodeInfo):
    """
    Class for storing the information returned by each environment on reset. It is parsed into a flat dictionary
    of arrays.

    """
    def parse(self):
        """
        Turn the appended episodes into a flat dictionary of arrays, one entry per key.

        Returns:
            A flat dictionary containing an array for every key of the episode information, with the
            environments one after the other.

        """
        if self._parsed is None:
            info = StepInfo(1, self.backend_name, self._device, vectorized=False)
            for entry in self._flat_entries():
                info.append(entry)

            if self._target_backend_name is not None:
                info = info.to_backend(self._target_backend_name, self._target_device)

            self._parsed = info.parse()

        return self._parsed

    def _create_instance(self, n_envs, vectorized):
        return ResetEpisodeInfo(n_envs, self.backend_name, self._device, vectorized=vectorized)


class ParametersEpisodeInfo(EpisodeInfo):
    """
    Class for storing the policy parameters drawn by an episodic agent at the start of each episode. It is parsed
    into a single array.

    """
    def parse(self):
        """
        Stack the stored policy parameters into a single array.

        Returns:
            The policy parameters of every episode along the first axis, with the environments one after the other,
            or ``None`` if no parameters are stored.

        """
        if self._parsed is None:
            entries = self._flat_entries()

            if len(entries) > 0:
                if self._target_backend_name is not None:
                    entries = [ArrayBackend.convert(entry, to=self._target_backend_name, backend=self._backend,
                                                    device=self._target_device) for entry in entries]
                    target = ArrayBackend.get_array_backend(self._target_backend_name)
                    self._parsed = target.from_list(entries, device=self._target_device)
                else:
                    self._parsed = self._backend.from_list(entries, device=self._device)

        return self._parsed

    def _create_instance(self, n_envs, vectorized):
        return ParametersEpisodeInfo(n_envs, self.backend_name, self._device, vectorized=vectorized)
