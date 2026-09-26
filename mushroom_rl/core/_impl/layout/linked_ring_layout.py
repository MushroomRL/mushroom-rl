from mushroom_rl.core.array_backend import ArrayBackend

from .episode_layout import EpisodeLayout
from .ring_layout import RingLayout


class LinkedRingLayout(RingLayout):
    """
    Class for the row structure of a circular buffer storing, for every position, the distance to the previous and to
    the next step of its episode.

    """
    def __init__(self, backend, max_size, device=None):
        """
        Constructor.

        Args:
            backend (str): the array backend;
            max_size (int): the capacity of the buffer;
            device (str, None): the device.

        """
        super().__init__(backend, max_size, device)
        self._prev_links = None
        self._next_links = None

        serialization = ArrayBackend.get_array_backend(backend).get_backend_serialization()
        self._add_save_attr(
            _prev_links=serialization,
            _next_links=serialization
        )

    def promote(self, last):
        """
        Build the layout of the same buffer storing the step links, which is this one.

        Args:
            last: the ``last`` flags of the buffer.

        Returns:
            This layout.

        """
        return self

    def orphans(self, positions):
        """
        Find the stored steps whose previous step is about to be overwritten.

        Args:
            positions (Array): the buffer position of every row about to be written.

        Returns:
            The positions of the stored steps whose previous step is about to be overwritten.

        """
        live = positions if self._full else positions[positions < self.size]
        continued = live[self._next_links[live] > 0]
        successors = (continued + self._next_links[continued]) % self._max_size
        return successors[(successors - self._write_head) % self._max_size >= len(positions)]

    def link(self, positions, continues, pairs, start):
        """
        Store the step links of the rows just written.

        Args:
            positions (Array): the buffer position of every written row;
            continues (Array): for every written row but the first, whether it continues the episode of the previous
                one;
            pairs (list): the ``(tail, head)`` pairs of the continued open episodes;
            start (int): the write head before the write.

        Returns:
            The positions of the open episode ends linked to their continuation.

        """
        backend = ArrayBackend.get_array_backend(self._backend)
        device = self._device
        n = len(positions)
        steps = backend.zeros(n, dtype=int, device=device)
        steps[1:] = continues * 1
        self._prev_links[positions] = steps
        steps = backend.zeros(n, dtype=int, device=device)
        steps[:-1] = continues * 1
        self._next_links[positions] = steps

        relinked = list()
        for tail, head in pairs:
            head_position = int(positions[head])
            if (tail - start) % self._max_size < n:
                self._prev_links[head_position] = self._max_size
            else:
                self._prev_links[head_position] = (head_position - tail) % self._max_size
                self._next_links[tail] = (head_position - tail) % self._max_size
                relinked.append(tail)
        return relinked

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
        def step(pos):
            distance = self._prev_links[pos]
            age = (pos - self._write_head) % self._max_size if self._full else pos
            return (pos - distance) % self._max_size, (distance > 0) & (distance <= age)

        return EpisodeLayout._linked_walk(anchors, n_hops, step)

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
        def step(pos):
            distance = self._next_links[pos]
            return (pos + distance) % self._max_size, distance > 0

        return EpisodeLayout._linked_walk(anchors, n_hops, step)

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
        positions, valid = self.walk_back(last, anchors, n_hops)
        return ~valid[:, -1] & (self._prev_links[positions[:, -1]] > 0)

    def segment_ends(self, last):
        ends = (last > 0) | (self._next_links[:len(last)] != 1)
        if len(ends) > 0:
            ends[-1] = True
        return ends

    def row_starts(self, last):
        return self._prev_links[:len(last)] == 0

    def time_order(self, last):
        backend = ArrayBackend.get_array_backend(self._backend)
        rows = backend.arange(0, self.size, device=self._device)
        age = self._age(rows)
        root = self._previous(last, rows, age)
        parent = root[root]
        while bool(backend.sum(parent != root) > 0):
            root, parent = parent, parent[parent]
        order = backend.argsort(age[root] * self._max_size + age)
        first = root[order] == order
        boundary_code = backend.zeros(len(order), dtype=self.dtype, device=self._device)
        boundary_code[first] = int(self.Boundary.CONTINUING)
        boundary_code[first & self.row_starts(last)[order]] = int(self.Boundary.FRESH)
        return order, boundary_code

    def to_backend(self, backend, device=None):
        layout = super().to_backend(backend, device)
        prev_links, next_links = ArrayBackend.convert(self._prev_links, self._next_links, to=backend, device=device)
        target = ArrayBackend.get_array_backend(backend)
        layout._prev_links, layout._next_links = target.copy(prev_links), target.copy(next_links)
        return layout

    @property
    def links(self):
        """
        The ``(prev, next)`` arrays of the distance from every buffer position to the previous and to the next step of
        its episode, 0 when there is none.

        """
        return self._prev_links, self._next_links

    def _contiguous_links(self, last):
        backend = ArrayBackend.get_array_backend(self._backend)
        device = self._device
        prev_links = backend.zeros(self._max_size, dtype=int, device=device)
        next_links = backend.zeros(self._max_size, dtype=int, device=device)
        size = self.size
        if size > 0:
            order = (backend.arange(0, size, device=device) + (self._write_head if self._full else 0)) % self._max_size
            open_rows = ~(last[order[:-1]] > 0) * 1
            prev_links[order[1:]] = open_rows
            next_links[order[:-1]] = open_rows
        return prev_links, next_links

    def _clear_rows(self):
        super()._clear_rows()
        backend = ArrayBackend.get_array_backend(self._backend)
        self._prev_links = backend.zeros(self._max_size, dtype=int, device=self._device)
        self._next_links = backend.zeros(self._max_size, dtype=int, device=self._device)

    def _follow_previous(self, rows):
        return (self._prev_links[rows] == 1) & (self._age(rows) > 0)

    def _previous(self, last, rows, age):
        backend = ArrayBackend.get_array_backend(self._backend)
        distance = self._prev_links[rows]
        follows = (distance > 0) & (distance <= age)
        return backend.where(follows, (rows - distance) % self._max_size, rows)
