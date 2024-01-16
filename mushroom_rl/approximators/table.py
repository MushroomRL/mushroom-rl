import numpy as np

from mushroom_rl.core.serialization import Serializable


class Table(Serializable):
    """
    Table regressor. Used for discrete state and action spaces.

    """
    def __init__(self, shape, initial_value=0., dtype=None):
        """
        Constructor.

        Args:
            shape (tuple): the shape of the tabular regressor.
            initial_value (float, 0.): the initial value for each entry of the
                tabular regressor.
            dtype ([int, float], None): the dtype of the table array.

        """
        self.table = np.ones(shape, dtype=dtype) * initial_value

        self._add_save_attr(table='numpy')

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
        """
        Args:
            x (int): index of the table to be filled;
            y (float): value to fill in the table.

        """
        self[x] = y

    def predict(self, *z):
        """
        Predict the output of the table given an input.

        Args:
            *z (list): list of input of the model. If the table is a Q-table,
            this list may contain states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;

        Returns:
            The table prediction.

        """
        if z[0].ndim == 1:
            z = [np.expand_dims(z_i, axis=0) for z_i in z]
        state = z[0]

        values = list()
        if len(z) == 2:
            action = z[1]
            for i in range(len(state)):
                val = self[state[i], action[i]]
                values.append(val)
        else:
            for i in range(len(state)):
                val = self[state[i], :]
                values.append(val)

        if len(values) == 1:
            return values[0]
        else:
            return np.array(values)

    @property
    def n_actions(self):
        """
        Returns:
            The number of actions considered by the table.

        """
        return self.table.shape[-1]

    @property
    def shape(self):
        """
        Returns:
            The shape of the table.

        """
        return self.table.shape

