from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.core.dataset import VectorizedDataset


class CollectDataset(Callback):
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """
    def __init__(self, initial_capacity=1000):
        """
        Constructor.

        Args:
            initial_capacity (int, 1000): number of transitions the accumulator is allocated for upfront. It
                grows geometrically from there, so this only bounds the number of reallocations; lower it for
                datasets with large observations.

        """
        self._dataset = None
        self._initial_capacity = initial_capacity

    def __call__(self, dataset):
        if self._dataset is None:
            self._dataset = dataset.copy()
            self._dataset.reserve(self._initial_capacity)
        else:
            capacity = self._dataset.capacity
            required = len(self._dataset) + len(dataset)
            if capacity is not None and required > capacity:
                self._dataset.reserve(max(2 * capacity, required))
            self._dataset += dataset

    def clean(self):
        self._dataset.clear()

    def get(self):
        if isinstance(self._dataset, VectorizedDataset):
            return self._dataset.flatten()
        return self._dataset
