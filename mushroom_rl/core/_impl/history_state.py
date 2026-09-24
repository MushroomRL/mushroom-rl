from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class HistoryContext:
    """
    The content of every history stream around the most recent step: the entries the window of the next step reuses
    and the entries the window of the most recent step reused.

    """
    def __init__(self, before_next, before_last):
        """
        Constructor.

        Args:
            before_next (dict): for each stream name, the entries preceding the next step, ``reach`` per environment;
            before_last (dict): for each stream name, the entries preceding the most recent step, ``reach`` per
                environment.

        """
        self.before_next = before_next
        self.before_last = before_last


class HistoryState(MushroomObject):
    """
    History stream entries attached to the rows of a flat dataset that start a segment continuing rows stored
    elsewhere: one entry per such row, keyed by row position, holding for every stream the entries that preceded the
    row. Kept in the agent backend.

    """
    def __init__(self, backend, device=None, positions=None, windows=None):
        """
        Constructor.

        Args:
            backend (str): the agent backend;
            device (str, None): the agent device;
            positions (Array, None): the row position of each entry; empty when ``None``;
            windows (dict, None): for each stream name, the entries of every row stacked along a leading axis; empty
                when ``None``.

        """
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device
        self._positions = self._array_backend.zeros(0, dtype=int, device=device) if positions is None else positions
        self._windows = dict() if windows is None else windows

        # numpy's save method cannot round-trip a dict, torch's can and relocates the device on load
        dict_serialization = 'torch' if self._array_backend.get_backend_serialization() == 'torch' else 'pickle'

        self._add_save_attr(
            _array_backend='primitive',
            _device='none',
            _positions=self._array_backend.get_backend_serialization(),
            _windows=dict_serialization
        )

    def __len__(self):
        return len(self._positions)

    def windows(self, name):
        """
        Get the stacked entries of one stream.

        Args:
            name (str): a stream name.

        Returns:
            The entries of that stream for every attached row, stacked along a leading axis aligned to
            :attr:`positions`, or ``None`` when the stream has no entry.

        """
        return self._windows.get(name)

    def get_view(self, index, n_rows):
        """
        Select the entries of the rows of a view of the dataset.

        Args:
            index (slice or Array): the rows selected by the view;
            n_rows (int): the number of rows of the dataset the view is taken from.

        Returns:
            The entries of the selected rows, keyed by their position in the view.

        """
        backend = self._array_backend
        if len(self) == 0:
            return self._wrap(self._positions, dict())
        if isinstance(index, slice):
            start, stop, step = index.indices(n_rows)
            if step == 1:
                keep = (self._positions >= start) & (self._positions < stop)
                return self._select(keep, self._positions[keep] - start)
            index = backend.arange(start, stop, step, device=self._device)
        index = backend.as_array(index, device=self._device)
        if index.dtype == bool:
            index = backend.where(index)[0]
        inverse = backend.zeros(n_rows, dtype=int, device=self._device) - 1
        inverse[index] = backend.arange(0, len(index), device=self._device)
        new_positions = inverse[self._positions]
        keep = new_positions >= 0
        return self._select(keep, new_positions[keep])

    def drop(self, rows, n_rows):
        """
        Copy the entries, leaving out the ones attached to the given rows.

        Args:
            rows (Array): the rows whose entries are left out;
            n_rows (int): the number of rows of the dataset.

        Returns:
            A new HistoryState holding the entries of every other row, at unchanged positions.

        """
        backend = self._array_backend
        if len(self) == 0 or len(rows) == 0:
            return self._wrap(self._positions, dict(self._windows))
        dropped = backend.zeros(n_rows, dtype=bool, device=self._device)
        dropped[backend.as_array(rows, device=self._device)] = True
        keep = ~dropped[self._positions]
        return self._select(keep, self._positions[keep])

    def concatenate(self, other, n_rows, stitched):
        """
        Join these entries with the ones of a dataset appended after this one.

        Args:
            other (HistoryState): the entries of the dataset appended after this one;
            n_rows (int): the number of rows of this dataset;
            stitched (bool): whether row 0 of the other dataset is the successor of the final row of this one, in which
                case its entry, if any, is dropped.

        Returns:
            The entries of the concatenated dataset.

        Raises:
            AssertionError: if the two datasets hold entries for different streams.

        """
        backend = self._array_backend
        other_positions = other._positions
        other_windows = other._windows
        if stitched and len(other) > 0:
            keep = other_positions != 0
            other_positions = other_positions[keep]
            other_windows = {name: window[keep] for name, window in other_windows.items()}
        if len(other_positions) == 0:
            return self._wrap(backend.copy(self._positions), dict(self._windows))
        if len(self) == 0:
            return self._wrap(other_positions + n_rows, other_windows)
        assert self._windows.keys() == other_windows.keys(), \
            "The concatenated datasets hold entries for different streams: " \
            f"{sorted(self._windows)} and {sorted(other_windows)}."

        positions = backend.concatenate_arrays([self._positions, other_positions + n_rows])
        windows = {name: backend.concatenate_arrays([window, other_windows[name]])
                   for name, window in self._windows.items()}
        return self._wrap(positions, windows)

    def clear(self):
        """
        Returns:
            The entries of the emptied dataset.

        """
        return self._wrap(self._positions[:0], dict())

    def to_backend(self, backend, device=None):
        """
        Convert the entries to another backend.

        Args:
            backend (str): the target backend name;
            device (str, None): the target device.

        Returns:
            The entries converted to the given backend.

        """
        target = ArrayBackend.get_array_backend(backend)
        positions = target.convert_to_backend(self._array_backend, self._positions, device)
        windows = {name: target.convert_to_backend(self._array_backend, window, device)
                   for name, window in self._windows.items()}
        return HistoryState(backend, device, positions, windows)

    @classmethod
    def from_context(cls, backend, device, history_context):
        """
        Build the entries of a single-environment dataset whose row 0 continues the most recent step.

        Args:
            backend (str): the agent backend;
            device (str, None): the agent device;
            history_context (HistoryContext): the stream content around the most recent step.

        Returns:
            The entries: one at row 0, holding the entries preceding the next step.

        """
        array_backend = ArrayBackend.get_array_backend(backend)
        positions = array_backend.zeros(1, dtype=int, device=device)
        return cls(backend, device, positions, {name: array_backend.copy(entries)[None] for name, entries in
                                                history_context.before_next.items()})

    @property
    def positions(self):
        """
        The row position of each entry.

        """
        return self._positions

    def _select(self, keep, positions):
        return self._wrap(positions, {name: window[keep] for name, window in self._windows.items()})

    def _wrap(self, positions, windows):
        return HistoryState(self._array_backend.get_backend_name(), self._device, positions, windows)

    def _post_load(self):
        self._device = self._array_backend.check_device(None)


class GridHistoryState(MushroomObject):
    """
    History stream content retained for a dataset collected from parallel environments: for every environment, the
    stream entries preceding its first row since the dataset was last cleared, to be attached to the flat datasets
    built from the grid. Kept in the agent backend.

    """
    def __init__(self, n_envs, backend, device=None):
        """
        Constructor.

        Args:
            n_envs (int): number of parallel environments;
            backend (str): the agent backend;
            device (str, None): the agent device.

        """
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device
        self._n_envs = n_envs
        self._slots = dict()

        # numpy's save method cannot round-trip a dict, torch's can and relocates the device on load
        dict_serialization = 'torch' if self._array_backend.get_backend_serialization() == 'torch' else 'pickle'

        self._add_save_attr(
            _array_backend='primitive',
            _device='none',
            _n_envs='primitive',
            _slots=dict_serialization
        )

    def reset(self, history_context, leftover, mask_backend):
        """
        Set the retained entries after a clear: the environments active in the leftover row get the entries preceding
        the most recent step, every other environment the entries preceding the next step.

        Args:
            history_context (HistoryContext, None): the stream content around the most recent step; nothing is
                retained when ``None``;
            leftover: boolean mask of the environments active in the leftover row;
            mask_backend (ArrayBackend): the backend of ``leftover``.

        """
        self._slots = dict()
        if history_context is not None:
            backend = self._array_backend
            leftover = backend.convert_mask(leftover[:self._n_envs], backend=mask_backend, device=self._device)
            for name, before_next in history_context.before_next.items():
                before_next = before_next[:self._n_envs]
                before_last = history_context.before_last[name][:self._n_envs]
                shape = (self._n_envs,) + (1,) * (len(before_next.shape) - 1)
                self._slots[name] = backend.where(leftover.reshape(shape), before_last, before_next)

    def emit(self, positions, envs, is_episode_start, positions_backend):
        """
        Build the entries of a flat dataset from the retained windows.

        Args:
            positions: the row positions, in the flat dataset, of the segment starts;
            envs: the environment each of those rows belongs to;
            is_episode_start: whether each of those rows starts an episode, in which case it gets no entry;
            positions_backend (ArrayBackend): the backend of the three arrays.

        Returns:
            The :class:`HistoryState` of the flat dataset.

        """
        backend = self._array_backend
        positions = backend.convert_to_backend(positions_backend, positions, self._device)
        envs = backend.convert_to_backend(positions_backend, envs, self._device)
        keep = backend.convert_to_backend(positions_backend, ~is_episode_start, self._device)
        positions, envs = positions[keep], envs[keep]
        windows = {name: slot[envs] for name, slot in self._slots.items()}
        return HistoryState(backend.get_backend_name(), self._device, positions, windows)

    def get_view(self, index, n_rows):
        return self.copy()

    def concatenate(self, other, n_rows, stitched):
        return self.copy()

    def clear(self):
        """
        Drop the retained entries.

        Returns:
            This object.

        """
        self._slots = dict()
        return self

    def to_backend(self, backend, device=None):
        state = GridHistoryState(self._n_envs, backend, device)
        state._slots = {name: state._array_backend.convert_to_backend(self._array_backend, slot, device)
                        for name, slot in self._slots.items()}
        return state

    def _post_load(self):
        self._device = self._array_backend.check_device(None)
