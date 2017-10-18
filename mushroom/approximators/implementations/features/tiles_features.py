import numpy as np


class TilesFeatures:

    def __init__(self, tiles):
        self._tiles = tiles
        self._size = 0

        for tiling in tiles:
            self._size += tiling.size

    def __call__(self, *args):
        out = np.empty(self._size)

        offset = 0
        for tiling in self._tiles:
            index = tiling(input) + offset

            if index is not None:
                out[index] = 1.0

            offset += tiling.size

        return out

    @property
    def size(self):
        return self._size