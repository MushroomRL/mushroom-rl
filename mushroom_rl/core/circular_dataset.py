from mushroom_rl.core.array_backend import ArrayBackend
from ._impl.layout import RingLayout
from mushroom_rl.core.dataset import Dataset


class CircularDataset(Dataset):
    """
    :class:`Dataset` variant backing a circular replay buffer of fixed capacity. Datasets are written at the write
    head, overwriting the oldest rows once the buffer is full. A written dataset continues the episodes left open by
    the previous one when it starts with continuing rows, one per open episode and in the same order; a dataset that
    continues nothing closes them, setting their stored ``last`` flag.

    """
    def __init__(self, dataset_info, max_size):
        """
        Constructor.

        Args:
            dataset_info (DatasetInfo): the static information used to build the dataset;
            max_size (int): the capacity of the buffer.

        """
        super().__init__(dataset_info, n_steps=max_size)

    def write(self, dataset):
        """
        Write a dataset at the write head, moving it forward. The dataset is converted to the backend and device of
        the buffer, and the episodes of joined datasets are reordered as by :meth:`Dataset.contiguous`.

        Args:
            dataset (Dataset): the dataset to write.

        Returns:
            The buffer position of every row of ``dataset``, in its order, the positions of the open episode ends of
            the previous write that the dataset continues, and the positions of the stored steps whose previous step
            was overwritten.

        Raises:
            ValueError: if the dataset holds more rows than the buffer, or continues a number of episodes different
                from the number left open.

        """
        if len(dataset) > self._layout.max_size:
            raise ValueError(f"Cannot write {len(dataset)} rows to a buffer of {self._layout.max_size} rows.")
        dataset, order = dataset._glued()
        dataset = dataset.to_backend(self._dataset_info.env_backend, device=self._dataset_info.env_device)
        n = len(dataset)
        max_size = self._layout.max_size
        backend = self._dataset_info.env_array_backend
        device = self._dataset_info.env_device
        start = self._layout.write_head
        positions = (backend.arange(0, n, device=device) + start) % max_size

        to_close, pairs = self._layout.pair(dataset._layout.pending_heads())
        if len(to_close) > 0:
            self.last[to_close] = True

        last = dataset._last_array()
        continues = dataset._layout.continues(last)
        adjacent = all((int(positions[head]) - tail) % max_size == 1 for tail, head in pairs)
        if self._layout.links is None and (not adjacent or dataset._layout.has_inner_break(last)):
            self._layout = self._layout.promote(self._last_array())

        orphans = self._layout.orphans(positions)

        self._write_rows(dataset)

        relinked = self._layout.link(positions, continues, pairs, start)

        self._layout.set_tails(dataset._layout.pending_tails(dataset.last), positions)

        if order is not None:
            order = ArrayBackend.convert(order, to=self._dataset_info.env_backend, device=device)
            written = positions
            positions = backend.zeros(n, dtype=int, device=device)
            positions[order] = written

        return positions, relinked, orphans

    def append_batch(self, other):
        self._append_rows(other)

    def history_cut(self, anchors, n_hops):
        """
        Check which anchors have a history window reaching a step that is no longer stored.

        Args:
            anchors (Array): the buffer position of each walk;
            n_hops (int): the number of steps to walk back.

        Returns:
            Whether walking back ``n_hops`` steps of the episode of each anchor reaches a step that is no longer stored.

        """
        return self._layout.history_cut(self._last_array(), anchors, n_hops)

    @property
    def is_circular(self):
        return True

    @property
    def max_size(self):
        """
        The capacity of the buffer.

        """
        return self._layout.max_size

    @property
    def write_head(self):
        """
        The buffer position the next row is written at.

        """
        return self._layout.write_head

    @property
    def full(self):
        """
        Whether the buffer has wrapped around.

        """
        return self._layout.full

    @property
    def size(self):
        """
        The number of rows stored.

        """
        return self._layout.size

    @property
    def links(self):
        """
        The ``(prev, next)`` arrays holding, for every buffer position, the distance to the previous and to the next
        step of its episode, 0 when there is none; ``None`` while every stored episode is written in consecutive
        positions, delimited by the stored ``last`` flags.

        """
        return self._layout.links

    def _write_rows(self, dataset):
        n_total = len(dataset)
        n = n_total
        max_size = self._layout.max_size
        head = self._layout.write_head

        if not self._layout.full:
            remaining = max_size - len(self)
            if n <= remaining:
                self.append_batch(dataset)
                self._layout.advance(n_total)
                return

            self.append_batch(dataset[:remaining])
            head = 0
            dataset = dataset[remaining:]
            n -= remaining

        columns = ['state', 'action', 'reward', 'next_state', 'absorbing', 'last']
        if self.is_stateful:
            columns += ['policy_state', 'policy_next_state']
        end = head + n
        if end <= max_size:
            for column in columns:
                getattr(self, column)[head:end] = getattr(dataset, column)
        else:
            first = max_size - head
            rest = n - first
            for column in columns:
                values = getattr(dataset, column)
                getattr(self, column)[head:] = values[:first]
                getattr(self, column)[:rest] = values[first:]
        self._layout.advance(n_total)

    def _walk_last(self):
        return self._last_array()

    def _create_layout(self, dataset_info, n_steps, n_envs):
        return RingLayout(dataset_info.env_backend, n_steps, dataset_info.env_device)
