from enum import IntFlag

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class EpisodeLayout(MushroomObject):
    """
    Base class for the row structure of a dataset: where each stored row breaks from the previous one, and the
    episodes left open to the datasets joined before and after it.

    """
    class Boundary(IntFlag):
        NONE = 0x0  # the row follows the previous row
        FRESH = 0x1  # the row starts an episode with no past
        CONTINUING = 0x2  # the row continues an episode whose past rows are not the previous row
        BLOCK_START = 0x4  # the row starts a block joined after the rows before it, with episodes to pair across
        BREAK = FRESH | CONTINUING  # the row does not follow the previous row

    def __init__(self, backend, shape=(), device=None, n_envs=None):
        """
        Constructor. Builds an empty layout.

        Args:
            backend (str): the array backend;
            shape (tuple, ()): the shape of the boundary column, capacity included;
            device (str, None): the device;
            n_envs (int, None): the number of parallel environments of each row, or ``None``.

        """
        self._backend = backend
        self._shape = shape
        self._device = device
        self._n_envs = n_envs
        self._first = int(self.Boundary.FRESH)
        self._open_heads = None
        self._open_tails = None
        self._n_joins = 0

        self._add_save_attr(
            _backend='primitive',
            _shape='primitive',
            _device='primitive',
            _n_envs='primitive',
            _first='primitive',
            _open_heads='primitive',
            _open_tails='primitive',
            _n_joins='primitive'
        )

    def append_batch(self, other, last):
        """
        Append the rows of another layout, pairing its continuing episodes with the open ones of this layout.

        Args:
            other (EpisodeLayout): the layout to append;
            last: the ``last`` flags of this layout.

        Returns:
            The layout holding the rows of both, which is this one unless its kind has to change, and whether the
            first appended row continues the final row.

        Raises:
            ValueError: if ``other`` continues a number of episodes different from the number left open.

        """
        stitched, glued, heads, tails = self._pair(other, last)
        n_joins = self._n_joins + other._n_joins + int(glued)
        layout = self._append(other, stitched, glued, last)
        layout._open_heads, layout._open_tails, layout._n_joins = heads, tails, n_joins
        return layout, stitched

    def concatenate(self, other, last):
        """
        Join this layout and another one, pairing the continuing episodes of ``other`` with the open ones of this
        layout.

        Args:
            other (EpisodeLayout): the layout that follows;
            last: the ``last`` flags of this layout.

        Returns:
            The joined layout, and whether the first row of ``other`` continues the final row.

        Raises:
            ValueError: if ``other`` continues a number of episodes different from the number left open.

        """
        stitched, glued, heads, tails = self._pair(other, last)
        result = self._joined(other, stitched, glued, last)
        result._open_heads, result._open_tails = heads, tails
        result._n_joins = self._n_joins + other._n_joins + int(glued)
        return result, stitched

    def clear(self, last=None):
        """
        Drop every row and the open episodes.

        Args:
            last (None): the ``last`` flags of the dropped rows; when given, the next appended row continues the final
                dropped row if its episode is open.

        """
        if last is not None and len(self) > 0:
            self._first = int(self.Boundary.FRESH if bool(last[-1]) else self.Boundary.CONTINUING)
        self._clear_rows()
        self._open_heads = None
        self._open_tails = None
        self._n_joins = 0

    def standalone_view(self, index, last):
        """
        Select rows as the layout of a standalone dataset.

        Args:
            index (slice or Array): the rows selected;
            last: the ``last`` flags of this layout.

        Returns:
            The layout of the selected rows, continuing no episode and leaving none open. A slice keeps the episode
            structure of its rows.

        """
        if isinstance(index, slice) and index.step in (None, 1):
            layout = self._standalone_slice(index, last)
        else:
            array_backend = ArrayBackend.get_array_backend(self._backend)
            starts = self.row_starts(last)[index]
            boundary = array_backend.zeros(len(starts), dtype=self.dtype, device=self._device)
            boundary[:] = int(self.Boundary.CONTINUING)
            boundary[starts] = int(self.Boundary.FRESH)
            layout = self._coded_from_array(boundary)
        layout._open_heads, layout._open_tails = tuple(), tuple()

        return layout

    def episodes_view(self, rows, last):
        """
        Select whole episodes as the layout of a standalone dataset.

        Args:
            rows (Array): the selected rows, increasing and covering whole episodes;
            last: the ``last`` flags of this layout.

        Returns:
            The layout of the selected rows, continuing no episode and leaving none open.

        """
        array_backend = ArrayBackend.get_array_backend(self._backend)
        boundary_code = array_backend.zeros(len(rows), dtype=self.dtype, device=self._device)
        boundary_code[:] = int(self.Boundary.CONTINUING)
        boundary_code[self.row_starts(last)[rows]] = int(self.Boundary.FRESH)
        linked = (rows[1:] == rows[:-1] + 1) & self._follow_previous(rows[1:])
        boundary_code[1:][linked] = int(self.Boundary.NONE)
        return self._coded_from_array(boundary_code, open_heads=tuple(), open_tails=tuple())

    def walk_back(self, last, anchors, n_hops):
        """
        Walk back from each anchor along its episode.

        Args:
            last: the flags of the final row of every stored segment;
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        return EpisodeLayout.walk_stream_back(last, anchors, n_hops)

    def walk_forward(self, last, anchors, n_hops):
        """
        Walk forward from each anchor along its episode.

        Args:
            last: the flags of the final row of every stored segment;
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        return EpisodeLayout.walk_stream_forward(last, anchors, n_hops)

    def pending_heads(self):
        """
        Returns:
            The rows continuing the open episodes of the dataset this one is appended to.

        """
        if self._open_heads is not None:
            return self._open_heads
        if len(self) > 0 and self._code(0) & self.Boundary.CONTINUING:
            return 0,
        return tuple()

    def pending_tails(self, last):
        """
        Get the rows whose episode the next appended dataset continues.

        Args:
            last: the ``last`` flags of the rows.

        Returns:
            The rows whose episode the next appended dataset continues.

        """
        if self._open_tails is not None:
            return self._open_tails
        if len(self) > 0 and not bool(last[-1]):
            return len(self) - 1,
        return tuple()

    @staticmethod
    def walk_stream_back(last, anchors, n_hops):
        """
        Walk back from each anchor along a stream of rows delimited by their ``last`` flags.

        Args:
            last: the ``last`` flags of the rows;
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        backend = ArrayBackend.get_array_backend_from(anchors)
        size = len(last)
        raw = anchors[:, None] - backend.arange(0, n_hops + 1, device=backend.get_device(anchors))[None, :]
        exists = (raw[:, 1:] >= 0) & ~(last[backend.clip(raw[:, 1:], 0, size - 1)] > 0)
        return EpisodeLayout._straight_walk(anchors, exists, -1, None)

    @staticmethod
    def walk_stream_forward(last, anchors, n_hops):
        """
        Walk forward from each anchor along a stream of rows delimited by their ``last`` flags.

        Args:
            last: the ``last`` flags of the rows;
            anchors (Array): the starting row of each walk;
            n_hops (int): the number of steps to walk.

        Returns:
            The ``(positions, valid)`` arrays of shape ``(len(anchors), n_hops + 1)``: column ``k`` holds the row
            reached after ``k`` steps, column 0 the anchor, and whether that step exists. A missing step holds the last
            row reached.

        """
        backend = ArrayBackend.get_array_backend_from(anchors)
        size = len(last)
        raw = anchors[:, None] + backend.arange(0, n_hops + 1, device=backend.get_device(anchors))[None, :]
        exists = (raw[:, 1:] < size) & ~(last[backend.clip(raw[:, :-1], 0, size - 1)] > 0)
        return EpisodeLayout._straight_walk(anchors, exists, 1, None)

    @property
    def continuing(self):
        """
        Whether row 0 continues the open episode of the dataset this one is appended to.

        """
        return len(self) > 0 and bool(self._code(0) & self.Boundary.CONTINUING)

    @property
    def first(self):
        """
        The boundary code of the first row appended to an empty layout.

        """
        return self._first

    @property
    def n_joins(self):
        """
        The number of joins with episodes to pair across them.

        """
        return self._n_joins

    @property
    def open_heads(self):
        """
        The rows continuing the open episodes of the dataset this one is appended to, or ``None`` for the default.

        """
        return self._open_heads

    @property
    def open_tails(self):
        """
        The rows whose episode the next appended dataset continues, or ``None`` for the default.

        """
        return self._open_tails

    @property
    def dtype(self):
        """
        The data type of the boundary codes.

        """
        return ArrayBackend.get_array_backend(self._backend).to_backend_dtype('int8')

    def _coded_from_array(self, boundary, open_heads=None, open_tails=None):
        raise NotImplementedError

    def _resized(self, n_rows):
        return (n_rows,) + tuple(self._shape[1:]) if self._backend != 'list' else ()

    def _copy_state(self, layout):
        layout._first, layout._n_joins = self._first, self._n_joins
        layout._open_heads, layout._open_tails = self._open_heads, self._open_tails
        return layout

    def _pair(self, other, last):
        n = len(self)
        if len(other) == 0:
            return False, False, self._open_heads, self._open_tails
        if n == 0:
            return False, False, other._open_heads, other._open_tails
        heads, tails = other.pending_heads(), self.pending_tails(last)
        if len(heads) > 0 and len(heads) != len(tails):
            raise ValueError(f"Cannot append a dataset continuing {len(heads)} episodes to a dataset ending with "
                             f"{len(tails)} open episodes.")
        stitched = len(heads) == 1 and heads[0] == 0 and tails[0] == n - 1
        glued = not stitched and (len(heads) > 0 or len(tails) > 0)
        other_tails = None if other._open_tails is None else tuple(t + n for t in other._open_tails)
        return stitched, glued, self._open_heads, other_tails

    def _mark_join(self, boundary, n, stitched, glued):
        if stitched:
            boundary.column()[n] = int(self.Boundary.NONE)
        if glued:
            boundary.column()[n] = int(boundary.column()[n]) | int(self.Boundary.BLOCK_START)

    @staticmethod
    def _straight_walk(anchors, exists, direction, modulo):
        # a walk moving one row per step: ``exists[:, k - 1]`` tells whether step ``k`` is possible from step ``k - 1``
        backend = ArrayBackend.get_array_backend_from(anchors)
        n_hops = exists.shape[1]
        steps = backend.arange(0, n_hops + 1, device=backend.get_device(anchors))
        if n_hops > 0:
            reached = backend.min(backend.where(exists, steps[None, -1:], steps[None, :-1]), dim=1)
        else:
            reached = anchors * 0
        valid = steps[None, :] <= reached[:, None]
        positions = anchors[:, None] + direction * backend.minimum(steps[None, :], reached[:, None])
        if modulo is not None:
            positions = positions % modulo
        return positions, valid

    @staticmethod
    def _linked_walk(anchors, n_hops, step):
        # a walk following stored links: ``step(pos)`` gives the next position and whether the step is possible
        backend = ArrayBackend.get_array_backend_from(anchors)
        pos = anchors
        active = backend.ones(len(anchors), dtype=bool, device=backend.get_device(anchors))
        positions = [pos]
        valid = [active]
        for _ in range(n_hops):
            target, exists = step(pos)
            active = active & exists
            pos = backend.where(active, target, pos)
            positions.append(pos)
            valid.append(active)
        return backend.stack(positions, 1), backend.stack(valid, 1)
