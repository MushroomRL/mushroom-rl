class Callback(object):
    """
    Interface for all basic callbacks. Implements a list in which it is possible
    to store data and methods to query and clean the content stored by the
    callback.

    """
    def __init__(self):
        """
        Constructor.

        """
        self._data_list = list()

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        raise NotImplementedError

    def get(self):
        """
        Returns:
             The current collected data as a list.

        """
        return self._data_list

    def clean(self):
        """
        Delete the current stored data list

        """
        self._data_list = list()
