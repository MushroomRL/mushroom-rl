import numpy as np


class Table:
    """
    Table regressor. Used for discrete state and action spaces.

    """
    def __init__(self, shape, initial_value=0., dtype=None):
        """
        Constructor.

        Args:
            shape (tuple): the shape of the tabular regressor.
            initial_value (float): the initial value for each entry of the
                tabular regressor.

        """
        self.__name__ = 'Table'
        self.table = np.ones(shape, dtype=dtype) * initial_value

    def __getitem__(self, args):
        if self.table.size == 1:
            return self.table[0]
        else:
            idx = tuple([
                a[0] if isinstance(a, np.ndarray) else a for a in args])

            return self.table[idx]

    def __setitem__(self, args, value):
        if self.table.size == 1:
            self.table[0] = value
        else:
            idx = tuple([
                a[0] if isinstance(a, np.ndarray) else a for a in args])
            self.table[idx] = value

    def fit(self, x, y):
        self[x] = y

    def predict(self, *z):
        if z[0].ndim == 1:
            z = [np.expand_dims(z_i, axis=0) for z_i in z]
        state = z[0]

        values = list()
        if len(z) == 2:
            action = z[1]
            for i in xrange(len(state)):
                val = self[state[i], action[i]]
                values.append(val)
        else:
            for i in xrange(len(state)):
                val = self[state[i], :]
                values.append(val)

        if len(values) == 1:
            return values[0]
        else:
            return np.array(values)

    @property
    def n_actions(self):
        return self.table.shape[-1]

    @property
    def shape(self):
        return self.table.shape

    def __str__(self):
        return self.__name__
