from mushroom_rl.core.array_backend import ArrayBackend

from .episode_layout import EpisodeLayout
from .stream_layout import StreamLayout


class RingLayout(StreamLayout):
    """
    Class for the row structure of a circular buffer whose stored episodes occupy consecutive positions: the write
    head, whether the buffer has wrapped, and the open episodes of the last write.

    """
    def __init__(self, backend, max_size, device=None):
        """
        Constructor.

        Args:
            backend (str): the array backend;
            max_size (int): the capacity of the buffer;
            device (str, None): the device.

        """
        super().__init__(backend, device=device)

        self._max_size = max_size
        self._write_head = 0
        self._full = False
        self._ring_tails = self._array_backend.zeros(0, dtype=int, device=self._device)

        self._add_save_attr(
            _max_size='primitive',
            _write_head='primitive',
            _full='primitive',
            _ring_tails=self._array_backend.get_backend_serialization()
        )

    def append_rows(self, other):
        if self._n_rows == 0 and len(other) > 0:
            self._head = int(self.Boundary.CONTINUING if other.continuing else self.Boundary.FRESH)
        self._n_rows += len(other)
        return self

    def pair(self, heads):
        """
        Pair the continuing episodes of a dataset about to be written with the open episodes of the last write.

        Args:
            heads (Array): the rows of the dataset continuing the open episodes, in order.

        Returns:
            The buffer positions of the open episodes to close, and the ``(tails, heads)`` pair of arrays of the
            continued ones: the buffer position of every continued open episode and the row continuing it, in order.

        Raises:
            ValueError: if the dataset continues a number of episodes different from the number left open.

        """
        no_rows = self._array_backend.zeros(0, dtype=int, device=self._device)
        to_close = no_rows
        pairs = no_rows, no_rows
        if self.size > 0:
            if len(heads) == 0:
                to_close = self._ring_tails
            elif len(heads) == len(self._ring_tails):
                pairs = self._ring_tails, heads
            else:
                raise ValueError(f"Cannot write a dataset continuing {len(heads)} episodes to a buffer with "
                                 f"{len(self._ring_tails)} open episodes.")
        return to_close, pairs

    def promote(self, last):
        """
        Build the layout of the same buffer storing the step links between the stored steps of every episode.

        Args:
            last: the ``last`` flags of the buffer.

        Returns:
            A :class:`LinkedRingLayout` holding the same buffer state.

        """
        layout = LinkedRingLayout(self._array_backend.get_backend_name(), self._max_size, self._device)
        layout._n_rows, layout._head = self._n_rows, self._head
        layout._write_head, layout._full, layout._ring_tails = self._write_head, self._full, self._ring_tails
        layout._prev_links, layout._next_links = layout._contiguous_links(last)
        return self._copy_state(layout)

    def orphans(self, positions):
        """
        Find the stored steps whose previous step is about to be overwritten.

        Args:
            positions (Array): the buffer position of every row about to be written.

        Returns:
            The positions of the stored steps whose previous step is about to be overwritten, none while the stored
            episodes occupy consecutive positions.

        """
        return self._array_backend.zeros(0, dtype=int, device=self._device)

    def advance(self, n):
        """
        Move the write head forward.

        Args:
            n (int): the number of rows written.

        """
        self._full = self._full or self._write_head + n >= self._max_size
        self._write_head = (self._write_head + n) % self._max_size

    def link(self, positions, continues, pairs, start):
        """
        Store the step links of the rows just written; there are none to store while the stored episodes occupy
        consecutive positions.

        Args:
            positions (Array): the buffer position of every written row;
            continues (Array): for every written row but the first, whether it continues the episode of the previous
                one;
            pairs (tuple): the ``(tails, heads)`` arrays of the continued open episodes, as returned by :meth:`pair`;
            start (int): the write head before the write.

        Returns:
            The positions of the open episode ends linked to their continuation.

        """
        return self._array_backend.zeros(0, dtype=int, device=self._device)

    def set_tails(self, tails, positions):
        """
        Record the open episodes of the rows just written.

        Args:
            tails (Array): the rows of the written dataset whose episode is open;
            positions (Array): the buffer position of every written row.

        """
        self._ring_tails = positions[tails]

    def walk_back(self, last, anchors, n_hops):
        """
        Walk back from each anchor along its episode, stopping at the oldest stored step.

        Args:
            last: the ``last`` flags of the buffer;
            anchors (Array): the starting buffer position of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        if not self._full:
            return EpisodeLayout.walk_stream_back(last, anchors, n_hops)

        backend = ArrayBackend.get_array_backend_from(anchors)
        raw = (anchors[:, None] - backend.arange(0, n_hops + 1, device=backend.get_device(anchors))[None, :]) \
            % self._max_size
        exists = (raw[:, :-1] != self._write_head) & ~(last[raw[:, 1:]] > 0)
        return EpisodeLayout._straight_walk(anchors, exists, -1, self._max_size)

    def walk_forward(self, last, anchors, n_hops):
        """
        Walk forward from each anchor along its episode, stopping at the newest stored step.

        Args:
            last: the ``last`` flags of the buffer;
            anchors (Array): the starting buffer position of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        if not self._full:
            return EpisodeLayout.walk_stream_forward(last, anchors, n_hops)

        backend = ArrayBackend.get_array_backend_from(anchors)
        raw = (anchors[:, None] + backend.arange(0, n_hops, device=backend.get_device(anchors))[None, :]) \
            % self._max_size
        exists = (raw != (self._write_head - 1) % self._max_size) & ~(last[raw] > 0)
        return EpisodeLayout._straight_walk(anchors, exists, 1, self._max_size)

    def history_cut(self, last, anchors, n_hops):
        """
        Check which anchors have a history window reaching a step that is no longer stored.

        Args:
            last: the ``last`` flags of the buffer;
            anchors (Array): the buffer position of each walk;
            n_hops (int): the number of steps to walk back.

        Returns:
            Whether walking back ``n_hops`` steps of the episode of each anchor reaches a step that is no longer stored.

        """
        if self._full:
            positions, valid = self.walk_back(last, anchors, n_hops)
            return ~valid[:, -1] & (positions[:, -1] == self._write_head)
        backend = ArrayBackend.get_array_backend_from(anchors)
        return backend.zeros(len(anchors), dtype=bool, device=backend.get_device(anchors))

    def segment_ends(self, last):
        ends = ArrayBackend.get_array_backend_from(last).copy(last)
        if len(ends) > 0:
            ends[-1] = True
            if self._full:
                ends[(self._write_head - 1) % self._max_size] = True
        return ends

    def segment_starts(self, last):
        raise NotImplementedError("The rows of this dataset are not stored as one stream, so their segment starts "
                                  "are unknown.")

    def row_starts(self, last):
        """
        Find the rows that start an episode.

        Args:
            last: the ``last`` flags of the buffer.

        Returns:
            The flags of the rows known to start an episode.

        """
        if not self._full:
            return super().row_starts(last)
        backend = ArrayBackend.get_array_backend_from(last)
        starts = backend.concatenate([last[-1:], last[:-1]]) > 0
        starts[self._write_head] = False
        return starts

    def standalone_view(self, index, last):
        if isinstance(index, slice) and index.step in (None, 1):
            backend = self._array_backend
            rows = backend.arange(0, len(self), device=self._device)[index]
            return self.episodes_view(rows, last)
        return super().standalone_view(index, last)

    def time_order(self, last):
        """
        Arrange the stored rows by episode, each episode from its oldest stored step on and the episodes by the age
        of their oldest stored step.

        Args:
            last: the ``last`` flags of the buffer.

        Returns:
            The buffer positions of the stored rows in that order, and the boundary code of each of them.

        """
        backend = self._array_backend
        rows = backend.arange(0, self.size, device=self._device)
        order = (rows + self._write_head) % self._max_size if self._full else rows
        starts = self.row_starts(last)[order]
        boundary_code = backend.zeros(len(order), dtype=self.dtype, device=self._device)
        boundary_code[starts] = int(self.Boundary.FRESH)
        if len(order) > 0 and not bool(starts[0]):
            boundary_code[0] = int(self.Boundary.CONTINUING)
        return order, boundary_code

    def to_backend(self, backend, device=None):
        layout = type(self)(backend, self._max_size, device)
        layout._n_rows, layout._head = self._n_rows, self._head
        layout._write_head, layout._full = self._write_head, self._full
        layout._ring_tails = self._converted_rows(self._ring_tails, backend, device)
        return self._converted_state(layout, backend, device)

    @property
    def max_size(self):
        """
        The capacity of the buffer.

        """
        return self._max_size

    @property
    def write_head(self):
        """
        The buffer position the next row is written at.

        """
        return self._write_head

    @property
    def full(self):
        """
        Whether the buffer has wrapped around.

        """
        return self._full

    @property
    def size(self):
        """
        The number of rows stored.

        """
        return self._max_size if self._full else self._write_head

    @property
    def links(self):
        """
        The ``(prev, next)`` step links of the buffer, ``None`` while every stored episode occupies consecutive
        positions.

        """
        return None

    @property
    def ring_tails(self):
        """
        Returns:
            The buffer positions of the open episodes of the last write.

        """
        return self._ring_tails

    def _clear_rows(self):
        super()._clear_rows()
        self._write_head = 0
        self._full = False
        self._ring_tails = self._array_backend.zeros(0, dtype=int, device=self._device)

    def _follow_previous(self, rows):
        return self._age(rows) > 0

    def _age(self, rows):
        return (rows - self._write_head) % self._max_size if self._full else rows


from .linked_ring_layout import LinkedRingLayout  # noqa: E402
