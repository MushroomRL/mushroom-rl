import numpy as np


class Tiles:
    def __init__(self, x_range, n_tiles, state_components=None):

        if isinstance(x_range[0], list):
            self._range = x_range
        else:
            self._range = [x_range]

        if isinstance(n_tiles, list):
            assert(len(n_tiles) == len(self._range))
            self._n_tiles = n_tiles
        else:
            self._n_tiles = [n_tiles]*len(self._range)

        self._state_components = state_components

        if self._state_components is not None:
            assert(len(self._state_components) == len(self._range))

        self._size = 1

        for s in self._n_tiles:
            self._size *= s

    def __call__(self, x):
        if self._state_components is not None:
            x = x[self._state_components]

        multiplier = 1
        tileIndex = 0

        for i, (r, N) in enumerate(zip(self._range, self._n_tiles)):
            if r[0] <= x[i] < r[1]:
                width = r[1] - r[0]
                componentIndex = int(np.floor(N * (x[i] - r[0]) / width))
                tileIndex += componentIndex * multiplier
                multiplier *= N
            else:
                tileIndex = None
                break

        return tileIndex

    @property
    def size(self):
        return self._size


