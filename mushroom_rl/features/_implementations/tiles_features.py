import numpy as np

from .features_implementation import FeaturesImplementation


class TilesFeatures(FeaturesImplementation):
    def __init__(self, tiles):

        if isinstance(tiles, list):
            self._tiles = tiles
        else:
            self._tiles = [tiles]
        self._size = 0

        for tiling in self._tiles:
            self._size += tiling.size

    def __call__(self, *args):
        x = self._concatenate(args)

        y = list()

        x = np.atleast_2d(x)
        for s in x:
            out = np.zeros(self._size)

            offset = 0
            for tiling in self._tiles:
                index = tiling(s)

                if index is not None:
                    out[index + offset] = 1.

                offset += tiling.size

            y.append(out)

        if len(y) == 1:
            y = y[0]
        else:
            y = np.array(y)

        return y

    def compute_indexes(self, *args):
        x = self._concatenate(args)

        y = list()

        x = np.atleast_2d(x)
        for s in x:
            out = list()

            offset = 0
            for tiling in self._tiles:
                index = tiling(s)

                if index is not None:
                    out.append(index + offset)

                offset += tiling.size

            y.append(out)

        if len(y) == 1:
            return y[0]
        else:
             return y

    @property
    def size(self):
        return self._size
