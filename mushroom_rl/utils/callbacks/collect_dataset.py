from mushroom_rl.utils.callbacks.callback import Callback

class CollectDataset(Callback):
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        self._data_list += dataset