from mushroom_rl.core.mushroom_object import MushroomObject


class Container(MushroomObject):
    """
    Base class of the columnar containers holding the rows of a dataset: an ordered list of equal-length columns,
    each independently shaped and typed, plus a shared length counter. One subclass per array backend, selected by
    the backend name through :meth:`create` and :meth:`from_array`.

    """
    _by_backend = dict()

    def __init_subclass__(cls, backend=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if backend is not None:
            Container._by_backend[backend] = cls

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def append(self, *values):
        """
        Append one row.

        Args:
            *values: one value per column.

        """
        raise NotImplementedError

    def append_batch(self, other):
        """
        Append the rows of another container of the same columns.

        Args:
            other (Container): the container whose rows are appended.

        """
        raise NotImplementedError

    def clear(self):
        """
        Drop every row.

        """
        raise NotImplementedError

    def reserve(self, capacity):
        """
        Ensure the container can hold at least ``capacity`` rows.

        Args:
            capacity (int): the number of rows.

        """
        raise NotImplementedError

    def compact(self, start):
        """
        Keep the rows from ``start`` on, moved to the front.

        Args:
            start (int): the first row kept.

        """
        raise NotImplementedError

    def get_view(self, index, copy=False):
        """
        Args:
            index (slice or Array): the rows selected;
            copy (bool, False): whether the view owns a copy of the selected rows.

        Returns:
            A container holding the selected rows.

        """
        raise NotImplementedError

    def column(self, index=None):
        """
        Args:
            index (int, None): the column; may be omitted for a single-column container.

        Returns:
            The stored rows of the column.

        """
        raise NotImplementedError

    def n_episodes(self, last_index, skip_incomplete=True):
        """
        Args:
            last_index (int): the column holding the ``last`` flags;
            skip_incomplete (bool, True): whether a trailing incomplete episode is left out.

        Returns:
            The number of episodes.

        """
        raise NotImplementedError

    @classmethod
    def create(cls, backend, shapes, dtypes, device=None, n_envs=None):
        """
        Build an empty container for the given backend.

        Args:
            backend (str): ``'numpy'``, ``'torch'`` or ``'list'``;
            shapes (list): the full shape of each column, capacity included;
            dtypes (list): the data type of each column;
            device (str, None): device of the columns, torch only;
            n_envs (int, None): number of parallel environments the columns are batched over, or ``None``.

        Returns:
            The container.

        """
        return cls._by_backend[backend]._allocate(shapes, dtypes, device, n_envs)

    @classmethod
    def from_array(cls, arrays, device=None, backend=None):
        """
        Wrap existing equal-length arrays into a container.

        Args:
            arrays (list): one array per column;
            device (str, None): device of the columns, torch only;
            backend (str, None): ``'numpy'``, ``'torch'`` or ``'list'``; required when called on this base class,
                ignored when called on a subclass.

        Returns:
            The container.

        """
        target = cls._by_backend[backend] if cls is Container else cls
        return target._from_array_impl(arrays, device)

    @property
    def data(self):
        """
        The stored rows of every column, as a list.

        """
        raise NotImplementedError

    @property
    def n_columns(self):
        raise NotImplementedError

    @property
    def n_envs(self):
        raise NotImplementedError

    @property
    def capacity(self):
        """
        The number of rows the container can hold, or ``None`` when it grows without bound.

        """
        raise NotImplementedError

    @classmethod
    def _allocate(cls, shapes, dtypes, device, n_envs):
        raise NotImplementedError

    @classmethod
    def _from_array_impl(cls, arrays, device):
        raise NotImplementedError
