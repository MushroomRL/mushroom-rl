from collections import deque

import numpy as np

from mushroom_rl.utils.plots import DataBuffer


class _preposProcessBuffer(DataBuffer):
    """
    This class implements a processing of the data before and after the data is appended.
    Useful if the user wants to do operations like means, standard deviations, ... of the data.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Args:
            *args (list) :positional arguments of DataBuffer;
            **kwargs (dict) : kwargs of the DataBuffer.
        """
        super().__init__(*args, **kwargs)

    def update(self, data):
        """
        Function that calls pre update processing and post update processing.
        Args:
            data (list) : Data to be processed and added to DataBuffer.
        """
        pre_update_res = self.pre_update(data)
        super().update(pre_update_res)
        self.pos_update()

    def pre_update(self, data):
        """
        Pre-processing method.
        Args:
            data (list) : Raw Data.
        """
        raise NotImplementedError

    def pos_update(self):
        """
        Post-processing method.
        """
        raise NotImplementedError


class EndOfEpisodeBuffer(_preposProcessBuffer):
    """
    This class adds the values until the end of the episode is recorded.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Args:
            args (str) :positional arguments of DataBuffer;
            kwargs (dict) : kwargs of the DataBuffer.
        """
        super().__init__(*args, **kwargs)

        self._accumulated_data = deque()  # Raw data
        self._accumulated_absorbing = deque()  # Absorbing states

    def pre_update(self, data):
        """
        Function that appends the data to the accumulation arrays and once it detects absorbing
        state calculates the sums of each set and appends to the base DataBuffer. Removes the
        samples already counted in to the DataBuffer.

        Args:
            data (list) : Raw data to be added.

        Returns:
            Returns the pre-processed data.
        """

        data = np.array(data)
        self._accumulated_data.extend(data[:, 0])
        self._accumulated_absorbing.extend(data[:, 1])
        indexes = self._get_absorbing_indexes()
        data = []

        first = 0
        temp_acc = np.array(list(self._accumulated_data))
        for i in indexes:
            data.append(np.sum(temp_acc[first:int(i)]))
            first = int(i)
        last = first + 1

        if indexes.size != 0:
            for i in range(last):
                self._accumulated_data.popleft()
                self._accumulated_absorbing.popleft()

        return data

    def _get_absorbing_indexes(self):
        """
        Get the indexes of the absorbing states within the accumulated data.

        Returns:
             Array of indexes.
        """
        indexes = np.argwhere(self._accumulated_absorbing)
        return indexes

    def pos_update(self):
        """
        Nothing happens in this implemation.
        """
        pass
