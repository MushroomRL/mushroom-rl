from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.core.dataset import VectorizedDataset


class CollectDataset(Callback):
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """
    def __init__(self):
        """
        Constructor.
        """
        self._dataset = None

    def __call__(self, dataset):
        if isinstance(dataset, VectorizedDataset):
            dataset = dataset.flatten()

        if self._dataset is None:
            self._dataset = dataset.copy()
        else:
            self._dataset += dataset

    def clean(self):
        self._dataset.clear()

    def get(self):
        return self._dataset
