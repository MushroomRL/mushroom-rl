class Callback(object):
    """
    Interface for all basic callbacks. Implements a list in which it is possible
    to store data and methods to query and clean the content stored by the
    callback.

    """
    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (Dataset): the samples to collect.

        """
        raise NotImplementedError

    def get(self):
        """
        Returns:
             The current collected data.

        """
        raise NotImplementedError

    def clean(self):
        """
        Delete the current stored data

        """
        raise NotImplementedError


class CallbackList(Callback):
    """
    Simple interface for callbacks storing a single list for data collection

    """
    def __init__(self):
        self._data_list = list()

    def get(self):
        """
        Returns:
             The current collected data.

        """
        return self._data_list

    def clean(self):
        """
        Delete the current stored data

        """
        self._data_list = list()

