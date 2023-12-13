import numpy as np


class Tiles:
    """
    Class implementing rectangular tiling. For each point in the state space,
    this class can be used to compute the index of the corresponding tile.

    """
    def __init__(self, x_range, n_tiles, state_components=None):
        """
        Constructor.

        Args:
            x_range (list): list of two-elements lists specifying the range of
                each state variable;
            n_tiles (list): list of the number of tiles to be used for each
                dimension.
            state_components (list, None): list of the dimensions of the input
                to be considered by the tiling. The number of elements must
                match the number of elements in ``x_range`` and ``n_tiles``.

        """
        if isinstance(x_range[0], list):
            self._range = x_range
        else:
            self._range = [x_range]

        if isinstance(n_tiles, list):
            assert(len(n_tiles) == len(self._range))

            self._n_tiles = n_tiles
        else:
            self._n_tiles = [n_tiles] * len(self._range)

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
        tile_index = 0

        for i, (r, N) in enumerate(zip(self._range, self._n_tiles)):
            if r[0] <= x[i] < r[1]:
                width = r[1] - r[0]
                component_index = int(np.floor(N * (x[i] - r[0]) / width))
                tile_index += component_index * multiplier
                multiplier *= N
            else:
                tile_index = None
                break

        return tile_index

    @staticmethod
    def generate(n_tilings, n_tiles, low, high, uniform=False):
        """
        Factory method to build ``n_tilings`` tilings of ``n_tiles`` tiles with
        a range between ``low`` and ``high`` for each dimension.

        Args:
            n_tilings (int): number of tilings, or -1 to compute the number
                             automatically;
            n_tiles (list): number of tiles for each tilings for each dimension;
            low (np.ndarray): lowest value for each dimension;
            high (np.ndarray): highest value for each dimension.
            uniform (bool, False): if True the displacement for each tiling will
                                   be w/n_tilings, where w is the tile width.
                                   Otherwise, the displacement will be
                                   k*w/n_tilings, where k=2i+1, where i is the
                                   dimension index.

        Returns:
            The list of the generated tiles.

        """
        assert len(n_tiles) == len(low) == len(high)
        assert n_tilings > 0 or n_tilings == -1

        if n_tilings == -1:
            n = np.max(n_tiles)
            d = np.size(low)  # space-dim
            m = np.max([np.ceil(np.log(4 * d) / np.log(2)),
                        np.ceil(np.log(n) / np.log(2))])
            n_tilings = int(m**2)

        # Min, max coord., side length of the state-space
        low = np.array(low, dtype=float)
        high = np.array(high, dtype=float)
        L = high - low

        # Unit shift displacement vector
        shift = 1 if uniform else 2 * np.arange(len(low)) + 1

        # Length of the sides of the tiles, l
        be = (n_tilings - 1) / n_tilings
        l = L / (np.array(n_tiles) - be)

        # Generate the list of tilings
        tilings = list()

        for i in range(n_tilings):
            # Shift vector
            v = (i * shift) % n_tilings

            # Min, max of the coordinates of the i-th tiling
            x_min = low + (-n_tilings + 1 + v) / n_tilings * l
            x_max = x_min + l * n_tiles

            # Rearrange x_min, x_max and append new tiling to the list
            x_range = [[x, y] for x, y in zip(x_min, x_max)]
            tilings.append(Tiles(x_range, n_tiles))

        return tilings

    @property
    def size(self):
        return self._size
