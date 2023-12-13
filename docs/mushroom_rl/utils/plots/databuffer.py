import pickle
from collections import deque


class DataBuffer(object):
    """
    This class represents the data of a certain variable and how it should be
    recorded.

    """
    def __init__(self, name, length=None):
        """
        Constructor.

        Args:
            name (str): name of the data buffer (each data buffer represents one
                variable);
            length (int, None): size of the data buffer queue. If length is
                ``None``, the buffer is unbounded.

        """
        self._buffer = deque(maxlen=length)
        self.name = name
        self._tracking_enabled = True

        assert isinstance(name, str), "Name of DataBuffer needs to be a string"

    @property
    def size(self):
        """
        Returns:
            The size of the queue.

        """
        return len(self._buffer)

    def update(self, data):
        """
        Append values to buffer if tracking enabled.

        Args:
            data (list): list of values to append.

        """
        if self._tracking_enabled:
            self._buffer.extend(data)

    def get(self):
        """
        Getter.

        Returns:
            Buffer queue.

        """
        return self._buffer

    def set(self, buffer):
        """
        Setter.

        Args:
            buffer (deque): the queue to be used as buffer.

        """
        self._buffer = buffer

    def reset(self):
        """
        Remove all the values from buffer.

        """
        self._buffer.clear()

    def enable_tracking(self, status):
        """
        Enable or disable tracking of data. If tracking is disabled, data is not
        stored in the buffer.

        Args:
            status (bool): whether to enable (True) or disable (False) tracking.

        """
        self._tracking_enabled = status

    def save(self, path):
        """
        Save the data buffer.

        """
        path = path + "/{}".format(self.name)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        """
        Load the data buffer.

        Returns:
            The loaded data buffer instance.

        """
        with open(path, "rb") as file:
            loaded_instance = pickle.load(file)

        return loaded_instance
