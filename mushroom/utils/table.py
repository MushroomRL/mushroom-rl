import numpy as np


class Table:
    """
    Table regressor. Used for discrete state and action spaces.

    """
    def __init__(self, shape, initial_value=0.):
        """
        Constructor.

        Args:
            shape (tuple): the shape of the tabular regressor.
            initial_value (float): the initial value for each entry of the
                tabular regressor.

        """
        self.__name__ = 'Table'
        self.table = np.ones(shape) * initial_value

    def __getitem__(self, args):
        idxs = self._get_index(args)

        return self.table[idxs]

    def __setitem__(self, args, value):
        idxs = self._get_index(args)
        self.table[idxs] = value

    def _get_index(self, args):
        if len(args) == 0:
            idxs = (0,)
        elif len(args) == 1:
            idxs = tuple(args[0].ravel())
        elif type(args[0]) is slice:
            idxs = (args[0],) * (
                len(self.table.shape)-1) + tuple(args[1].astype(int))
        elif type(args[1]) is slice:
            idxs = tuple(args[0].astype(int)) + (args[1],)
        else:
            idxs = tuple(np.concatenate((args[0].astype(int),
                                         args[1].astype(int))))

        return idxs

    def fit(self, x, y):
        self[x] = y

    def predict(self, *z):
        state = z[0]
        table = list()
        if len(z) == 2:
            action = z[1]
            for i in xrange(len(state)):
                val = self[state[i], action[i]]
                table.append(val)
        else:
            for i in xrange(len(state)):
                val = self[state[i], :]
                table.append(val)

        return np.array(table)

    @property
    def shape(self):
        return self.table.shape

    def __str__(self):
        return self.__name__
