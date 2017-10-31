import numpy as np


class TilesFeatures:

    def __init__(self, tiles):

        if isinstance(tiles, list):
            self._tiles = tiles
        else:
            self._tiles = [tiles]
        self._size = 0

        for tiling in self._tiles:
            self._size += tiling.size

    def __call__(self, *args):
        if len(args) > 1:
            x = np.concatenate(args, axis=0)
        else:
            x = args[0]

        out = np.zeros(self._size)

        offset = 0
        for tiling in self._tiles:
            index = tiling(x)

            if index is not None:
                out[index + offset] = 1.

            offset += tiling.size

        return out

    @property
    def size(self):
        return self._size
