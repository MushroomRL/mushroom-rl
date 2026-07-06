from copy import deepcopy

from mushroom_rl.core.mushroom_object import MushroomObject


class ListDataset(MushroomObject):
    """
    Growable storage using plain Python lists. It grows without pre-allocation (which allows collecting episodes of
    unbounded/infinite horizon) and can hold ragged or non-array content. Holds an ordered list of equal-length columns
    (one list per column).

    """
    def __init__(self, n_columns, n_envs=None):
        self._columns = [list() for _ in range(n_columns)]
        self._n_envs = n_envs

        super().__init__()

        self._add_all_save_attr()

    @classmethod
    def create_new_instance(cls, dataset=None):
        """
        Creates an empty instance of the dataset and populates essential data structures.

        Args:
            dataset (ListDataset, None): a template dataset to be used to create the new instance.

        Returns:
            A new empty instance of the dataset.

        """
        new_dataset = cls.__new__(cls)

        new_dataset._columns = None
        new_dataset._n_envs = dataset._n_envs if dataset is not None else None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, arrays):
        dataset = cls.create_new_instance()

        dataset._columns = [list(array) for array in arrays]

        return dataset

    def __len__(self):
        return len(self._columns[0])

    def append(self, *values):
        for column, value in zip(self._columns, values):
            column.append(deepcopy(value if self._n_envs is None else value[:self._n_envs]))

    def append_batch(self, other):
        for column, other_column in zip(self._columns, other.columns):
            column.extend(deepcopy(other_column))

    def clear(self):
        self._columns = [list() for _ in range(self.n_columns)]

    def get_view(self, index, copy=False):
        view = self.create_new_instance(self)

        if isinstance(index, (int, slice)):
            view._columns = [column[index] for column in self._columns]
        else:
            view._columns = [[column[i] for i in index] for column in self._columns]

        if copy:
            view._columns = deepcopy(view._columns)

        return view

    def __getitem__(self, index):
        return tuple(column[index] for column in self._columns)

    def __add__(self, other):
        result = self.create_new_instance(self)

        result._columns = [column + other_column
                           for column, other_column in zip(self._columns, other.columns)]

        return result

    def column(self, index=None):
        if index is None:
            assert self.n_columns == 1
            index = 0
        return self._columns[index]

    @property
    def columns(self):
        return list(self._columns)

    @property
    def n_columns(self):
        return len(self._columns)

    def n_episodes(self, last_index):
        last = self.column(last_index)

        if len(last) == 0:
            return 0

        n_episodes = sum(1 for value in last if value)

        if not last[-1]:
            n_episodes += 1

        return n_episodes

    def _add_all_save_attr(self):
        self._add_save_attr(
            _columns='pickle'
        )
