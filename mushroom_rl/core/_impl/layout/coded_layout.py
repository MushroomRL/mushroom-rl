from mushroom_rl.core.array_backend import ArrayBackend

from ..containers import Container
from .episode_layout import EpisodeLayout


class CodedLayout(EpisodeLayout):
    """
    Class for the row structure of a dataset storing the boundary code of every row.

    """
    def __init__(self, backend, shape=(), device=None, n_envs=None):
        """
        Constructor. Builds an empty layout.

        Args:
            backend (str): the array backend;
            shape (tuple, ()): the shape of the boundary column, capacity included;
            device (str, None): the device;
            n_envs (int, None): the number of parallel environments of each row, or ``None``.

        """
        super().__init__(backend, shape, device, n_envs)
        self._boundary = Container.create(backend, [shape], [self.dtype], device, n_envs)

        self._add_save_attr(
            _boundary='mushroom'
        )

    def __len__(self):
        return len(self._boundary)

    def append(self, value=None):
        """
        Append one row.

        Args:
            value (None): its boundary code; by default the first-row code on an empty layout and ``NONE`` otherwise.

        """
        if value is None:
            value = self._first if len(self) == 0 else int(self.Boundary.NONE)
        self._boundary.append(value)

    def append_rows(self, other):
        """
        Append the rows of another layout with their boundary codes.

        Args:
            other (EpisodeLayout): the layout to append.

        Returns:
            This layout.

        """
        if len(other) > 0:
            self._boundary.append_batch(other._rows())
        return self

    def join_rows(self, other):
        """
        Join the rows of this layout and of another one, without pairing their episodes.

        Returns:
            A layout with the rows of this layout followed by the rows of ``other``, with their boundary codes.

        """
        result = CodedLayout.from_container(self._boundary + other._rows(), self._backend,
                                            self._resized(len(self) + len(other)), self._device, self._n_envs)
        result._n_joins = self._n_joins + other._n_joins
        return result

    def reserve(self, capacity):
        """
        Ensure the layout can hold at least ``capacity`` rows.

        Args:
            capacity (int): the number of rows.

        """
        self._boundary.reserve(capacity)

    def compact(self, start):
        """
        Keep the rows from ``start`` on, moved to the front.

        Args:
            start (int): the first row kept.

        """
        self._boundary.compact(start)

    def view(self, index):
        """
        Copy the selected rows.

        Args:
            index (slice or Array): the rows selected.

        Returns:
            A layout with a copy of the selected rows and their boundary codes.

        """
        return CodedLayout.from_container(self._boundary.get_view(index, copy=True), self._backend, self._shape,
                                          self._device, self._n_envs)

    def column(self):
        """
        Returns:
            The stored boundary codes.

        """
        return self._boundary.column()

    def array(self):
        """
        Returns:
            The stored boundary codes as an array.

        """
        return ArrayBackend.get_array_backend(self._backend).as_array(self._boundary.column(), device=self._device)

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
            ends[:-1][self.array()[1:] > 0] = True
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
        boundary = self.array()
        after = backend.where((last[:-1] > 0) | (boundary[1:] > 0))[0] + 1
        positions = backend.concatenate([backend.zeros(1, dtype=int, device=device), after])
        kind = boundary[positions]
        return positions, kind & int(self.Boundary.CONTINUING) == 0

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
        boundary = self.array()
        return (boundary & int(self.Boundary.FRESH) > 0) | ((boundary & int(self.Boundary.BREAK) == 0) & after_last)

    def continues(self, last):
        """
        Find the rows that continue the episode of the previous row.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            For every row but the first, whether it continues the episode of the previous row.

        """
        return (self.array()[1:] & int(self.Boundary.BREAK) == 0) & ~(last[:-1] > 0)

    def has_inner_break(self, last):
        """
        Check whether an open episode is followed by a row that does not continue it.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            Whether a row after the first breaks from a previous row whose episode is open.

        """
        if len(self) <= 1:
            return False
        return bool(((self.array()[1:] > 0) & ~(last[:-1] > 0)).any())

    def glue(self, last):
        """
        Reorder the rows so that every episode continued across a join follows its past rows. The open episodes of each
        joined block pair, in order, with the continuing episodes of the next block; the other rows keep their order.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            For every new row the row it comes from, the layout of the reordered rows, and the rows whose past now
            precedes them.

        Raises:
            ValueError: if a block continues a number of episodes different from the number left open by the previous
            one.

        """
        codes = self.array()
        backend = ArrayBackend.get_array_backend_from(codes)
        device = self._device
        n = len(self)

        is_start = codes > 0
        is_start[0] = True
        starts = backend.where(is_start)[0]
        ends = backend.concatenate([starts[1:] - 1, backend.zeros(1, dtype=int, device=device) + n - 1])
        lengths = ends - starts + 1
        block = backend.cumsum((codes[starts] & int(self.Boundary.BLOCK_START) > 0) * 1)
        is_head = codes[starts] & int(self.Boundary.CONTINUING) > 0
        is_tail = ~(last[ends] > 0)
        n_chunks, n_blocks = len(starts), int(block[-1]) + 1

        pred = backend.zeros(n_chunks, dtype=int, device=device) - 1
        for k in range(1, n_blocks):
            heads = backend.where((block == k) & is_head)[0]
            tails = backend.where((block == k - 1) & is_tail)[0]
            if len(heads) > 0 and len(heads) != len(tails):
                raise ValueError(f"Cannot glue a block continuing {len(heads)} episodes to a block ending with "
                                 f"{len(tails)} open episodes.")
            pred[heads] = tails[:len(heads)]
        has_pred = pred >= 0

        offset = backend.zeros(n_chunks, dtype=int, device=device)
        root = backend.arange(0, n_chunks, device=device)
        for k in range(1, n_blocks):
            linked = backend.where((block == k) & has_pred)[0]
            offset[linked] = offset[pred[linked]] + lengths[pred[linked]]
            root[linked] = root[pred[linked]]
        has_succ = backend.zeros(n_chunks, dtype=bool, device=device)
        has_succ[pred[has_pred]] = True
        chain_ends = backend.where(~has_succ)[0]
        total = backend.zeros(n_chunks, dtype=int, device=device)
        total[root[chain_ends]] = offset[chain_ends] + lengths[chain_ends]
        total = backend.where(has_pred, backend.zeros(n_chunks, dtype=int, device=device), total)
        new_start = (backend.cumsum(total) - total)[root] + offset

        chunk_of_row = backend.cumsum(is_start * 1) - 1
        new_row = new_start[chunk_of_row] + backend.arange(0, n, device=device) - starts[chunk_of_row]
        order = backend.zeros(n, dtype=int, device=device)
        order[new_row] = backend.arange(0, n, device=device)

        new_codes = backend.zeros(n, dtype=self.dtype, device=device)
        unlinked = backend.where(~has_pred)[0]
        new_codes[new_start[unlinked]] = codes[starts[unlinked]] & int(self.Boundary.BREAK)
        heads = tuple(int(i) for i in new_start[(block == 0) & is_head])
        tails = tuple(int(i) for i in new_row[ends[(block == n_blocks - 1) & is_tail]])
        layout = CodedLayout.from_array(new_codes, self._backend, device, open_heads=heads, open_tails=tails)
        layout._first = self._first

        return order, layout, starts[has_pred]

    def to_backend(self, backend, device=None):
        """
        Convert the layout to another backend.

        Args:
            backend (str): the target array backend;
            device (str, None): the target device.

        Returns:
            A new layout holding the same rows and open episodes in the given backend.

        """
        target = ArrayBackend.get_array_backend(backend)
        column = target.zeros(len(self), dtype=target.to_backend_dtype('int8'), device=device)
        column[:] = ArrayBackend.convert(self._boundary.column(), to=backend,
                                         backend=ArrayBackend.get_array_backend(self._backend), device=device)
        layout = CodedLayout.from_array(column, backend, device)
        return self._copy_state(layout)

    @classmethod
    def from_array(cls, boundary, backend, device=None, open_heads=None, open_tails=None):
        """
        Build a layout from an array of boundary codes.

        Args:
            boundary (Array): the boundary code of every row;
            backend (str): the array backend of ``boundary``;
            device (str, None): the device of ``boundary``;
            open_heads (tuple, None): the rows continuing the open episodes of the dataset this one is appended to; by
                default row 0 when it continues;
            open_tails (tuple, None): the rows whose episode the next appended dataset continues; by default the final
                row when its episode is open.

        Returns:
            The layout.

        """
        layout = cls.from_container(Container.from_array([boundary], device=device, backend=backend), backend,
                                    device=device)
        layout._shape = layout._resized(len(layout))
        layout._open_heads, layout._open_tails = open_heads, open_tails
        return layout

    @classmethod
    def from_container(cls, container, backend, shape=(), device=None, n_envs=None):
        """
        Build a layout holding the boundary codes of a container.

        Args:
            container (Container): the single-column container of the boundary codes;
            backend (str): the array backend of ``container``;
            shape (tuple, ()): the shape of the boundary column, capacity included;
            device (str, None): the device of ``container``;
            n_envs (int, None): the number of parallel environments of each row, or ``None``.

        Returns:
            The layout.

        """
        layout = cls.__new__(cls)
        EpisodeLayout.__init__(layout, backend, shape, device, n_envs)
        layout._boundary = container
        layout._add_save_attr(_boundary='mushroom')
        return layout

    def _append(self, other, stitched, glued, last):
        n = len(self)
        self.append_rows(other)
        self._mark_join(self._boundary, n, stitched, glued)
        return self

    def _joined(self, other, stitched, glued, last):
        n = len(self)
        return CodedLayout.from_container(self._boundary + other._rows(), self._backend, self._resized(n + len(other)),
                                          self._device, self._n_envs)._marked(n, stitched, glued)

    def _marked(self, n, stitched, glued):
        self._mark_join(self._boundary, n, stitched, glued)
        return self

    def _clear_rows(self):
        self._boundary.clear()

    def _standalone_slice(self, index, last):
        layout = self.view(index)
        start, stop, _ = index.indices(len(self))
        if max(start, stop) > start and start > 0 and self._code(start) & int(self.Boundary.BREAK) == 0:
            layout._boundary.column()[0] = int(self.Boundary.FRESH if bool(last[start - 1])
                                               else self.Boundary.CONTINUING)
        if self._n_joins > 0:
            layout = CodedLayout.from_array(layout.array() & int(self.Boundary.BREAK), self._backend, self._device)
        return layout

    def _follow_previous(self, rows):
        return self.array()[rows] & int(self.Boundary.BREAK) == 0

    def _coded_from_array(self, boundary, open_heads=None, open_tails=None):
        return CodedLayout.from_array(boundary, self._backend, self._device, open_heads=open_heads,
                                      open_tails=open_tails)

    def _code(self, row):
        return int(self._boundary.column()[row])

    def _rows(self):
        return self._boundary
