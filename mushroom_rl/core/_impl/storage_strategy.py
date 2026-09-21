from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class StorageStrategy(MushroomObject):
    """
    Base class for the strategies describing how the rows of a dataset are linked into trajectories.

    Every method taking ``last`` receives the stored ``last`` column of the dataset holding the strategy, as a numpy
    or torch array; its length is the number of rows.

    """
    def segment_ends(self, last):
        """
        Args:
            last: the stored ``last`` flags.

        Returns:
            A new array of the same length and data type, True where a row is the final row of a segment: at every
            ``last``, wherever the following row is not the row's successor, and on the final row.

        """
        raise NotImplementedError

    def segment_starts(self, last):
        """
        Args:
            last: the stored ``last`` flags.

        Returns:
            The positions of the rows starting a segment and, aligned to them, whether each start is an episode start
            (the environment was reset before the row) rather than the continuation of a segment stored elsewhere.

        """
        raise NotImplementedError

    def get_view(self, index, last):
        """
        Args:
            index (slice or array): the rows selected by the view;
            last: the stored ``last`` flags of the dataset the view is taken from.

        Returns:
            The strategy of the view.

        """
        raise NotImplementedError

    def stitches(self, other, last, other_last):
        """
        Args:
            other (StorageStrategy): the strategy of the dataset appended after this one;
            last: the stored ``last`` flags of this dataset;
            other_last: the stored ``last`` flags of the other dataset.

        Returns:
            Whether row 0 of the other dataset is the successor of the final row of this one.

        """
        raise NotImplementedError

    def concatenate(self, other, last, other_last):
        """
        Args:
            other (StorageStrategy): the strategy of the dataset appended after this one;
            last: the stored ``last`` flags of this dataset;
            other_last: the stored ``last`` flags of the other dataset.

        Returns:
            The strategy of the concatenated dataset.

        """
        raise NotImplementedError

    def append_batch(self, other, last, other_last):
        """
        Append the rows of another dataset in place, as in :meth:`concatenate`.

        Args:
            other (StorageStrategy): the strategy of the appended dataset;
            last: the stored ``last`` flags of this dataset before the append;
            other_last: the stored ``last`` flags of the appended dataset.

        Returns:
            The strategy of the dataset after the append.

        """
        raise NotImplementedError

    def append(self, last):
        """
        Extend the strategy by one row appended after the stored ones.

        Args:
            last: the stored ``last`` flags before the append.

        Returns:
            The strategy of the dataset after the append.

        """
        raise NotImplementedError

    def reserve(self, capacity):
        """
        Ensure the strategy can describe at least ``capacity`` rows.

        Args:
            capacity (int): the number of rows.

        """
        raise NotImplementedError

    def clear(self, last):
        """
        Args:
            last: the stored ``last`` flags of the dataset being emptied.

        Returns:
            The strategy of the emptied dataset.

        """
        raise NotImplementedError

    def to_backend(self, backend, device=None):
        """
        Args:
            backend (str): the target backend name;
            device (str, None): the target device.

        Returns:
            The strategy converted to the given backend.

        """
        raise NotImplementedError

    @property
    def continues(self):
        """
        Whether row 0 continues a row stored before this dataset.

        """
        raise NotImplementedError


class UntrackedRows(StorageStrategy):
    """
    Strategy of a dataset whose rows are not stored as one stream and whose links are not tracked: the segment ends
    are the stored ``last`` flags and every structural operation leaves the strategy unchanged.

    """
    def segment_ends(self, last):
        return ArrayBackend.get_array_backend_from(last).copy(last)

    def segment_starts(self, last):
        raise NotImplementedError

    def get_view(self, index, last):
        return self

    def stitches(self, other, last, other_last):
        return False

    def concatenate(self, other, last, other_last):
        return self

    def append_batch(self, other, last, other_last):
        return self

    def append(self, last):
        return self

    def reserve(self, capacity):
        pass

    def clear(self, last):
        return self

    def to_backend(self, backend, device=None):
        return self

    @property
    def continues(self):
        return False


class ContiguousRows(StorageStrategy):
    """
    Strategy of a dataset whose consecutive rows are consecutive steps of one stream, so that every non-adjacency
    coincides with a ``last``. It holds two scalars: the stream id of row 0 and whether row 0 continues a row stored
    before it.

    """
    def __init__(self, first_id=0, continues=False):
        """
        Constructor.

        Args:
            first_id (int, 0): the stream id of row 0;
            continues (bool, False): whether row 0 continues the row of id ``first_id - 1``, stored elsewhere.

        """
        assert first_id > 0 or not continues

        self._first_id = first_id
        self._continues = continues

        self._add_save_attr(
            _first_id='primitive',
            _continues='primitive'
        )

    def segment_ends(self, last):
        backend = ArrayBackend.get_array_backend_from(last)
        ends = backend.copy(last)
        if len(ends) > 0:
            ends[-1] = True
        return ends

    def segment_starts(self, last):
        backend = ArrayBackend.get_array_backend_from(last)
        device = backend.get_device(last)
        n = len(last)
        if n == 0:
            return backend.zeros(0, dtype=int, device=device), backend.zeros(0, dtype=bool, device=device)
        after_last = backend.where(last[:-1] > 0)[0] + 1
        positions = backend.concatenate([backend.zeros(1, dtype=int, device=device), after_last])
        is_episode_start = backend.ones(len(positions), dtype=bool, device=device)
        is_episode_start[0] = not self._continues
        return positions, is_episode_start

    def get_view(self, index, last):
        n = len(last)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            if step == 1:
                continues = self._continues if start == 0 else not bool(last[start - 1])
                return ContiguousRows(self._first_id + start, continues)
            backend = ArrayBackend.get_array_backend_from(last)
            index = backend.arange(start, stop, step, device=backend.get_device(last))
        return self.promote(last).get_view(index, last)

    def stitches(self, other, last, other_last):
        n = len(last)
        if n == 0 or len(other_last) == 0:
            return False
        if isinstance(other, ContiguousRows):
            return other._continues and other._first_id == self._first_id + n
        return int(other.prev_id[0]) == self._first_id + n - 1

    def concatenate(self, other, last, other_last):
        n = len(last)
        if n == 0:
            return other.copy()
        if isinstance(other, ContiguousRows):
            stitched = self.stitches(other, last, other_last)
            closed = bool(last[-1])
            if stitched or (closed and not other._continues):
                return ContiguousRows(self._first_id, self._continues)
        return self.promote(last).concatenate(other, last, other_last)

    def append_batch(self, other, last, other_last):
        return self.concatenate(other, last, other_last)

    def append(self, last):
        return self

    def reserve(self, capacity):
        pass

    def clear(self, last):
        n = len(last)
        if n == 0:
            return self
        return ContiguousRows(self._first_id + n, not bool(last[-1]))

    def to_backend(self, backend, device=None):
        return ContiguousRows(self._first_id, self._continues)

    def promote(self, last):
        """
        Args:
            last: the stored ``last`` flags.

        Returns:
            The equivalent :class:`PointerRows`, with the stream ids made explicit.

        """
        backend = ArrayBackend.get_array_backend_from(last)
        device = backend.get_device(last)
        n = len(last)
        ids = backend.arange(self._first_id, self._first_id + n, device=device)
        prev = ids - 1
        if n > 0:
            prev[1:][last[:-1] > 0] = -1
            if not self._continues:
                prev[0] = -1
        return PointerRows(ids, prev, backend.get_backend_name(), device)

    @property
    def first_id(self):
        """
        The stream id of row 0.

        """
        return self._first_id

    @property
    def continues(self):
        """
        Whether row 0 continues a row stored before this dataset.

        """
        return self._continues


class PointerRows(StorageStrategy):
    """
    Strategy of a dataset whose rows carry an explicit stream id and the id of their predecessor, so that rows
    ``i - 1`` and ``i`` are consecutive steps iff ``prev[i] == id[i - 1]``. A predecessor of ``-1`` marks an episode
    start. Rows are never appended one at a time; blocks are concatenated.

    """
    def __init__(self, ids, prev, backend, device=None):
        """
        Constructor.

        Args:
            ids: the stream id of every row, int64;
            prev: the stream id of every row's predecessor, ``-1`` at an episode start, int64;
            backend (str): the backend of the two arrays;
            device (str, None): their device, torch only.

        """
        self._ids = ids
        self._prev = prev
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device

        self._add_save_attr(
            _ids=self._array_backend.get_backend_serialization(),
            _prev=self._array_backend.get_backend_serialization(),
            _array_backend='primitive',
            _device='primitive'
        )

    def segment_ends(self, last):
        backend = ArrayBackend.get_array_backend_from(last)
        ends = backend.copy(last)
        if len(ends) > 0:
            ends[:-1][self._prev[1:] != self._ids[:-1]] = True
            ends[-1] = True
        return ends

    def segment_starts(self, last):
        backend = self._array_backend
        device = self._device
        n = len(self._ids)
        if n == 0:
            return backend.zeros(0, dtype=int, device=device), backend.zeros(0, dtype=bool, device=device)
        jumps = backend.where(self._prev[1:] != self._ids[:-1])[0] + 1
        positions = backend.concatenate_arrays([backend.zeros(1, dtype=int, device=device), jumps])
        return positions, self._prev[positions] == -1

    def get_view(self, index, last):
        return PointerRows(self._ids[index], self._prev[index], self._array_backend.get_backend_name(), self._device)

    def stitches(self, other, last, other_last):
        if len(self._ids) == 0 or len(other_last) == 0:
            return False
        if isinstance(other, ContiguousRows):
            other = other.promote(other_last)
        return int(other.prev_id[0]) == int(self._ids[-1])

    def concatenate(self, other, last, other_last):
        if isinstance(other, ContiguousRows):
            other = other.promote(other_last)
        backend = self._array_backend
        return PointerRows(backend.concatenate_arrays([self._ids, other.row_id]),
                           backend.concatenate_arrays([self._prev, other.prev_id]), backend.get_backend_name(),
                           self._device)

    def append_batch(self, other, last, other_last):
        if isinstance(other, ContiguousRows):
            other = other.promote(other_last)
        self._ids = self._array_backend.concatenate_arrays([self._ids, other.row_id])
        self._prev = self._array_backend.concatenate_arrays([self._prev, other.prev_id])
        return self

    def append(self, last):
        raise NotImplementedError

    def reserve(self, capacity):
        pass

    def clear(self, last):
        return PointerRows(self._ids[:0], self._prev[:0], self._array_backend.get_backend_name(), self._device)

    def to_backend(self, backend, device=None):
        target = ArrayBackend.get_array_backend(backend)
        ids = target.convert_to_backend(self._array_backend, self._ids, device)
        prev = target.convert_to_backend(self._array_backend, self._prev, device)
        return PointerRows(ids, prev, backend, device)

    @property
    def row_id(self):
        """
        The stream id of every row.

        """
        return self._ids

    @property
    def prev_id(self):
        """
        The stream id of every row's predecessor, ``-1`` at an episode start.

        """
        return self._prev

    @property
    def continues(self):
        return len(self._ids) > 0 and int(self._prev[0]) != -1

    @property
    def array_backend(self):
        return self._array_backend


class GridRows(StorageStrategy):
    """
    Strategy of a dataset collected from parallel environments, whose rows are one grid row per step with one entry
    per environment. It keeps the stream ids of the environments and stamps them onto the flat datasets built from
    the grid: every call to :meth:`stamp` numbers the selected entries as the next rows of the stream, in the order
    :meth:`~mushroom_rl.core.array_backend.ArrayBackend.pack_padded_sequence` lays them out.

    """
    def __init__(self, n_envs, backend, device=None):
        """
        Constructor.

        Args:
            n_envs (int): number of parallel environments;
            backend (str): the backend of the masks and flags handed to :meth:`stamp`;
            device (str, None): their device, torch only.

        """
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device

        self._next_id = 0
        self._last_id = self._array_backend.zeros(n_envs, dtype=int, device=device) - 1
        self._last_last = self._array_backend.ones(n_envs, dtype=bool, device=device)

        self._add_save_attr(
            _array_backend='primitive',
            _device='primitive',
            _next_id='primitive',
            _last_id=self._array_backend.get_backend_serialization(),
            _last_last=self._array_backend.get_backend_serialization()
        )

    def stamp(self, mask, last):
        """
        Number the grid entries selected by ``mask`` as the next rows of the stream.

        Args:
            mask: boolean array of shape ``(steps, n_envs)`` marking the entries to stamp;
            last: the stored ``last`` flags of the grid, of the same shape.

        Returns:
            The :class:`PointerRows` of the flat dataset holding the selected entries, and the environment each of
            its rows belongs to.

        """
        backend = self._array_backend
        device = self._device
        mask = backend.as_array(mask, device=device)
        counts = backend.sum(mask, dim=0)
        n = int(backend.sum(counts))
        ids = backend.arange(self._next_id, self._next_id + n, device=device)
        env_of_row = backend.repeat(backend.arange(0, len(counts), device=device), counts)
        if n == 0:
            return PointerRows(ids, backend.copy(ids), backend.get_backend_name(), device), env_of_row

        offsets = backend.cumsum(counts) - counts
        first = ids - self._next_id == offsets[env_of_row]
        flat_last = backend.pack_padded_arrays(backend.as_array(last, device=device), mask) > 0
        shifted_last = backend.concatenate_arrays([backend.zeros(1, dtype=bool, device=device), flat_last[:-1]])
        prev_last = backend.where(first, self._last_last[env_of_row], shifted_last)
        prev = backend.where(first, self._last_id[env_of_row], ids - 1)
        prev = backend.where(prev_last, backend.zeros(n, dtype=int, device=device) - 1, prev)

        active = counts > 0
        tail = (offsets + counts - 1)[active]
        self._last_id[active] = ids[tail]
        self._last_last[active] = flat_last[tail]
        self._next_id += n

        return PointerRows(ids, prev, backend.get_backend_name(), device), env_of_row

    def segment_ends(self, last):
        return ArrayBackend.get_array_backend_from(last).copy(last)

    def segment_starts(self, last):
        raise NotImplementedError

    def get_view(self, index, last):
        return self.copy()

    def stitches(self, other, last, other_last):
        return False

    def concatenate(self, other, last, other_last):
        return self.copy()

    def append_batch(self, other, last, other_last):
        return self

    def append(self, last):
        return self

    def reserve(self, capacity):
        pass

    def clear(self, last):
        return self

    def to_backend(self, backend, device=None):
        strategy = GridRows(len(self._last_id), backend, device)
        strategy._next_id = self._next_id
        strategy._last_id = strategy._array_backend.convert_to_backend(self._array_backend, self._last_id, device)
        strategy._last_last = strategy._array_backend.convert_to_backend(self._array_backend, self._last_last, device)
        return strategy

    @property
    def continues(self):
        return False

    @property
    def array_backend(self):
        return self._array_backend
