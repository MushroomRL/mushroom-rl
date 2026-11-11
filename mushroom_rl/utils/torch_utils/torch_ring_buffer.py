import torch


class TorchRingBuffer:
    """
    Class for a batch of fixed-length rolling histories, where every new entry overwrites the oldest one.

    """
    def __init__(self, length, batch_size, shape, dtype=torch.float, device=None):
        """
        Constructor.

        Args:
            length (int): The number of entries kept for every element of the batch.
            batch_size (int): The number of elements of the batch.
            shape (tuple): The shape of a single entry of an element.
            dtype (torch.dtype, torch.float): The data type of the entries.
            device (torch.device, None): The device the entries are stored on, or ``None`` for the default one.

        """
        self._buffer = torch.zeros((length, batch_size, *shape), dtype=dtype, device=device)
        self._batch_range = torch.arange(batch_size, device=self._buffer.device)
        self._head = 0

    def push(self, value):
        """
        Appends an entry for every element of the batch, overwriting the oldest one.

        Args:
            value (torch.tensor): The new entries, of shape ``(batch_size, *shape)``.

        """
        self._buffer[self._head] = value
        self._head = (self._head + 1) % len(self._buffer)

    def get(self, index):
        """
        Returns:
            The entry of every element of the batch at the given position, counted from the newest entry, ``0``
            being the newest, of shape ``(batch_size, *shape)``. ``index`` is an int, or a tensor with one position
            per element; each position is smaller than the length of the buffer.

        """
        return self._buffer[(self._head - 1 - index) % len(self._buffer), self._batch_range]

    def clear(self, batch_indices):
        """
        Zeroes every entry of the given elements of the batch.

        Args:
            batch_indices (torch.tensor): The elements to clear.

        """
        self._buffer.index_fill_(1, batch_indices, 0)
