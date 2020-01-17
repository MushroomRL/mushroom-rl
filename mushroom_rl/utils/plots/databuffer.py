import pickle
from collections import deque


class DataBuffer(object):
    """
    This class represents the data of a certain variable and how it should be recorded.

    """
    def __init__(self, name, length=None, tracking_enabled=True):
        """
        Constructor.

        Args:
            name (str): Name of the DataBuffer(each DataBuffer represents one variable);
            length (int): Size of the DataBuffer. If length is None the buffer will
                infinitly accumulate data. If the ammount of samples surpasses the size,
                data is taken out of the buffer as a queue.
            tracking_enabled (bool): Value of initialization of attribute _tracking_enabled.
                This flag determines whether the DataBuffer should include the given value to
                itself. This is usefull when plotting, and there is no need to always be appending
                information. Allows for faster script.
        """

        self._buffer = deque(maxlen=length)
        self.name = name
        self._tracking_enabled = tracking_enabled

        assert isinstance(name, str), "Name of DataBuffer needs to be a string"

    @property
    def size(self):
        """
        Size of queue

        Returns:
            Size of the queue
        """
        return len(self._buffer)

    def update(self, data):
        """
        Append values to buffer if tracking enabled.

        Args:
            data (list) : list of values to append.
        """
        if self._tracking_enabled:
            self._buffer.extend(data)

    def get(self):
        """
        Get buffer.

        Returns:
            Buffer deque.

        """
        return self._buffer

    def set(self, buffer):
        """
        Set buffer.

        Args:
            buffer (deque)
        """
        self._buffer = buffer

    def reset(self):
        """
        Remove all the values from buffer.
        """
        self._buffer.clear()

    def tracking_state(self, state):
        """
        Set _tracking_state.

        Args:
            state (bool): Flag value of tracking state.
        """
        self._tracking_enabled = state

    def save(self, path):
        path = path + "/{}".format(self.name)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load(self, path):
        with open(path, "rb") as file:
            loaded_instance = pickle.load(file)
        return loaded_instance
