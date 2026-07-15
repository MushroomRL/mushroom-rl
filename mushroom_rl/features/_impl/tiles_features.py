import numpy as np

from mushroom_rl.features.features import Features


class TilesFeatures(Features):
    def __init__(self, feature_list=None, backend='numpy'):
        self._tiles = self._as_list(feature_list)

        self._compute_size()

        super().__init__(backend, internal_backend='numpy')

        self._add_save_attr(_tiles='mushroom')

    def compute_indexes(self, *args):
        """
        Compute the index of the active tile of every tiling, for each sample of the raw input.

        Args:
            *args (list): the raw input.

        Returns:
            The index of the active tile of every tiling, with shape ``(n_samples, n_tilings)``, or ``-1`` for the
            tilings the sample falls outside of.

        """
        x = np.atleast_2d(self._convert_input(args))

        return self._tile_indexes(x)

    @property
    def size(self):
        return self._size

    def _compute(self, x):
        x = np.atleast_2d(x)

        indexes = self._tile_indexes(x)

        y = np.zeros((x.shape[0], self._size))
        rows, tilings = np.nonzero(indexes >= 0)
        y[rows, indexes[rows, tilings]] = 1.

        if y.shape[0] == 1:
            return y[0]

        return y

    def _tile_indexes(self, x):
        indexes = np.empty((x.shape[0], len(self._tiles)), dtype=int)

        offset = 0
        for i, tiling in enumerate(self._tiles):
            index = tiling(x)
            indexes[:, i] = np.where(index < 0, -1, index + offset)

            offset += tiling.size

        return indexes

    def _post_load(self):
        self._compute_size()

    def _compute_size(self):
        self._size = 0

        for tiling in self._tiles:
            self._size += tiling.size
