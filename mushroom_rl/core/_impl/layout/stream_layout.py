from mushroom_rl.core.array_backend import ArrayBackend

from ..containers import Container
from .episode_layout import EpisodeLayout
from .coded_layout import CodedLayout


class StreamLayout(EpisodeLayout):
    """
    Class for the row structure of consecutive steps of one stream: every row follows the previous one, and episodes
    are delimited by their ``last`` flags alone.

    """
    def __init__(self, backend, shape=(), device=None, n_envs=None):
        """
        Constructor. Builds an empty layout.

        Args:
            backend (str): the array backend;
            shape (tuple, ()): the shape of the boundary column the layout would store, capacity included;
            device (str, None): the device;
            n_envs (int, None): the number of parallel environments of each row, or ``None``.

        """
        super().__init__(backend, shape, device, n_envs)
        self._n_rows = 0
        self._head = int(self.Boundary.FRESH)

        self._add_save_attr(
            _n_rows='primitive',
            _head='primitive'
        )

    def __len__(self):
        return self._n_rows

    def append(self):
        """
        Append one row.

        """
        if self._n_rows == 0:
            self._head = self._first
        self._n_rows += 1

    def append_rows(self, other):
        """
        Append the rows of another layout with their boundary codes.

        Args:
            other (EpisodeLayout): the layout to append.

        Returns:
            The layout holding the rows of both, which is this one unless its kind has to change.

        """
        if len(other) == 0:
            return self
        if isinstance(other, StreamLayout) and len(self) == 0:
            self._n_rows, self._head = other._n_rows, other._head
            return self
        return self.coded().append_rows(other)

    def reserve(self, capacity):
        """
        Ensure the layout can hold at least ``capacity`` rows.

        Args:
            capacity (int): the number of rows.

        """
        if len(self._shape) > 0 and capacity > self._shape[0]:
            self._shape = (capacity,) + self._shape[1:]

    def view(self, index):
        """
        Copy the selected rows.

        Args:
            index (slice or Array): the rows selected.

        Returns:
            A layout with a copy of the selected rows and their boundary codes.

        """
        if isinstance(index, slice) and index.step in (None, 1):
            start, stop, _ = index.indices(self._n_rows)
            layout = StreamLayout(self._backend, self._shape, self._device, self._n_envs)
            layout._n_rows = max(stop - start, 0)
            layout._head = self._head if start == 0 else int(self.Boundary.NONE)
            return layout
        return CodedLayout.from_container(self._rows().get_view(index, copy=True), self._backend, self._shape,
                                          self._device, self._n_envs)

    def segment_ends(self, last):
        """
        Find the final row of every stored segment.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            The flags of the final row of every stored segment: every ``last``, every row not followed by its successor,
            and the final row.

        """
        ends = ArrayBackend.get_array_backend_from(last).copy(last)
        if len(ends) > 0:
            ends[-1] = True
        return ends

    def segment_starts(self, last):
        """
        Find the first row of every stored segment.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            The first row of every stored segment, and whether it starts an episode.

        """
        backend = ArrayBackend.get_array_backend_from(last)
        device = backend.get_device(last)
        if len(last) == 0:
            return backend.zeros(0, dtype=int, device=device), backend.zeros(0, dtype=bool, device=device)
        after = backend.where(last[:-1] > 0)[0] + 1
        positions = backend.concatenate([backend.zeros(1, dtype=int, device=device), after])
        is_episode_start = backend.ones(len(positions), dtype=bool, device=device)
        is_episode_start[0] = self._head != self.Boundary.CONTINUING
        return positions, is_episode_start

    def row_starts(self, last):
        """
        Find the rows that start an episode.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            The flags of the rows that start an episode.

        """
        backend = ArrayBackend.get_array_backend_from(last)
        after_last = backend.concatenate([backend.zeros(1, dtype=bool, device=backend.get_device(last)), last[:-1] > 0])
        if len(after_last) > 0:
            after_last[0] = self._head == self.Boundary.FRESH
        return after_last

    def continues(self, last):
        """
        Find the rows that continue the episode of the previous row.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            For every row but the first, whether it continues the episode of the previous row.

        """
        return ~(last[:-1] > 0)

    def has_inner_break(self, last):
        """
        Check whether an open episode is followed by a row that does not continue it.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            Whether a row after the first breaks from a previous row whose episode is open.

        """
        return False

    def coded(self):
        """
        Build a layout storing the boundary code of every row.

        Returns:
            A :class:`CodedLayout` holding the same rows and open episodes.

        """
        rows = self._rows() if self._n_rows > 0 else None
        container = Container.create(self._backend, [self._shape], [self.dtype], self._device, self._n_envs)
        if rows is not None:
            container.append_batch(rows)
        layout = CodedLayout.from_container(container, self._backend, self._shape, self._device, self._n_envs)
        return self._copy_state(layout)

    def to_backend(self, backend, device=None):
        """
        Convert the layout to another backend.

        Args:
            backend (str): the target array backend;
            device (str, None): the target device.

        Returns:
            A new layout holding the same rows and open episodes in the given backend.

        """
        layout = StreamLayout(backend, device=device)
        layout._shape = layout._resized(len(self))
        layout._n_rows, layout._head = self._n_rows, self._head
        return self._copy_state(layout)

    @classmethod
    def from_rows(cls, n_rows, backend, device=None, continuing=False):
        """
        Build the layout of ``n_rows`` consecutive steps.

        Args:
            n_rows (int): the number of rows;
            backend (str): the array backend;
            device (str, None): the device;
            continuing (bool, False): whether row 0 continues the open episode of the dataset this one is appended to.

        Returns:
            The layout.

        """
        layout = cls(backend, device=device)
        layout._shape = layout._resized(n_rows)
        layout._n_rows = n_rows
        layout._head = int(cls.Boundary.CONTINUING if continuing else cls.Boundary.FRESH)
        return layout

    def _append(self, other, stitched, glued, last):
        if self._follows(other, stitched, glued, last):
            if len(self) == 0:
                self._head = other._head
            self._n_rows += len(other)
            return self
        return self.coded()._append(other, stitched, glued, last)

    def _joined(self, other, stitched, glued, last):
        n = len(self)
        if self._follows(other, stitched, glued, last):
            result = StreamLayout(self._backend, self._resized(n + len(other)), self._device, self._n_envs)
            result._n_rows = n + len(other)
            result._head = self._head if n > 0 else other._head
            return result
        return CodedLayout.from_container(self._rows() + other._rows(), self._backend, self._resized(n + len(other)),
                                          self._device, self._n_envs)._marked(n, stitched, glued)

    def _clear_rows(self):
        self._n_rows = 0

    def _standalone_slice(self, index, last):
        layout = self.view(index)
        start, stop, _ = index.indices(len(self))
        if max(start, stop) > start and start > 0:
            layout._head = int(self.Boundary.FRESH if bool(last[start - 1]) else self.Boundary.CONTINUING)
        return layout

    def _follow_previous(self, rows):
        backend = ArrayBackend.get_array_backend_from(rows)
        return backend.ones(len(rows), dtype=bool, device=backend.get_device(rows))

    def _coded_from_array(self, boundary, open_heads=None, open_tails=None):
        return CodedLayout.from_array(boundary, self._backend, self._device, open_heads=open_heads,
                                      open_tails=open_tails)

    def _code(self, row):
        return self._head if row == 0 else int(self.Boundary.NONE)

    def _rows(self):
        array_backend = ArrayBackend.get_array_backend(self._backend)
        codes = array_backend.zeros(self._n_rows, dtype=self.dtype, device=self._device)
        if self._n_rows > 0:
            codes[0] = self._head
        return Container.from_array([codes], device=self._device, backend=self._backend)

    def _follows(self, other, stitched, glued, last):
        # whether the rows of ``other`` can be appended without storing a boundary code: ``other`` stores none either,
        # and its first row follows the final row here or starts an episode right after one ended
        if not isinstance(other, StreamLayout) or glued:
            return False
        if len(self) == 0 or len(other) == 0 or stitched:
            return True
        return other._head == self.Boundary.FRESH and bool(last[-1])
